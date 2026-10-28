import json
from pathlib import Path
import threading
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
from g1_deploy.core import CommandLatch, NEUTRAL_GRIP
from g1_deploy.transport import RobotTransport
from g1_deploy.wla_client import load_config,policy_images,execution_indices,camera_snapshot,predict_with_hold,submit_command,submit_on_schedule,check_chunk_progress

ROOT=Path(__file__).resolve().parents[1]

class WLATransportTests(unittest.TestCase):
    def setUp(self):
        self.command={'arm_q':[.1]*14,'arm_tau':[2.0]*14,'base_cmd':[.1,0,0,0,0,0,.74]}

    def test_position_hold_preserves_last_target_and_gravity(self):
        latch=CommandLatch(1,gripper_mode='position');latch.update(self.command,[2.1,4.2],10)
        for t in (10.2,10.8):
            c,g=latch.sample(t,True);self.assertEqual(c['arm_q'],[.1]*14);self.assertEqual(c['arm_tau'],[2.0]*14)
            np.testing.assert_equal(g,[2.1,4.2])
        c,g=latch.sample(11.1,True)
        self.assertEqual(c['base_cmd'][:3],[0,0,0]);self.assertEqual(c['arm_q'],[.1]*14)
        np.testing.assert_equal(g,[2.1,4.2])
        with self.assertRaises(RuntimeError):latch.update(self.command,[2,2],12)

    def test_control_mode_still_neutralizes_triggers(self):
        latch=CommandLatch(.25);latch.update(self.command,[0,0,10,1],10)
        _,g=latch.sample(11,True);np.testing.assert_equal(g,NEUTRAL_GRIP)

    def test_no_position_to_trigger_confusion(self):
        for mode,wrong in [('position',[10,0,10,0]),('controls',[2,4])]:
            with self.assertRaises(ValueError):CommandLatch(1,gripper_mode=mode).update(self.command,wrong,0)

    def test_readonly_has_both_grippers_but_no_publishers(self):
        channel=types.ModuleType('unitree_sdk2py.core.channel');channel.ChannelFactoryInitialize=lambda *a:None
        channel.ChannelPublisher=lambda *a:self.fail('Read-only publisher created')
        subscribed=[]
        class Sub:
            def __init__(self,topic,kind):subscribed.append(topic)
            def Init(self,*a):pass
            def Close(self):pass
        channel.ChannelSubscriber=Sub
        low=types.ModuleType('unitree_sdk2py.idl.unitree_hg.msg.dds_');low.LowState_=object
        grip=types.ModuleType('unitree_sdk2py.idl.unitree_go.msg.dds_');grip.MotorStates_=object
        with patch.dict('sys.modules',{'unitree_sdk2py.core.channel':channel,'unitree_sdk2py.idl.unitree_hg.msg.dds_':low,'unitree_sdk2py.idl.unitree_go.msg.dds_':grip}):
            robot=RobotTransport(load_config(ROOT/'config/wla-cup.example.json'),execute=False)
            robot.set_command({},[])
            self.assertEqual(subscribed,['rt/lowstate','rt/dex1/left/state','rt/dex1/right/state'])
            self.assertEqual(robot.publishers,[]);self.assertIsNone(robot.thread);robot.close()

    def test_absolute_gripper_publisher_is_rate_limited(self):
        r=RobotTransport.__new__(RobotTransport)
        r.config={'max_state_age_s':1,'gripper_command_mode':'position','gripper_min':0,'gripper_max':5.5,'gripper_speed_per_s':1}
        r.lock=threading.Lock();r.closed=threading.Event();r.state_time=time.monotonic();r.gripper_time=np.full(2,r.state_time)
        r.gripper_q=np.array([2.,4.]);r.remote_stop=False;r.latch=CommandLatch(1,gripper_mode='position')
        r.latch.update(self.command,[3.,3.],time.monotonic());r.String=lambda data:types.SimpleNamespace(data=data)
        body=[];grips=[[],[]]
        class BodyPub:
            def Write(self,msg):body.append(json.loads(msg.data));return True
        class GripPub:
            def __init__(self,i):self.i=i
            def Write(self,msg):
                grips[self.i].append(msg.cmds[0].q)
                if self.i==1 and len(grips[1])>=4:r.closed.set()
                return True
        r.body_pub=BodyPub();r.grip_pubs=[GripPub(0),GripPub(1)]
        r.grip_messages=[types.SimpleNamespace(cmds=[types.SimpleNamespace(q=0.)]) for _ in range(2)]
        r._publish()
        self.assertIsNone(r.latch.fault)
        self.assertTrue(all(c['arm_q']==[.1]*14 and c['arm_tau']==[2.]*14 for c in body))
        self.assertGreater(grips[0][0],2);self.assertLess(grips[1][0],4)
        self.assertLessEqual(grips[0][-1]-2,.021);self.assertLessEqual(4-grips[1][-1],.021)

    def inference_fixture(self, delay=.32):
        c={'max_camera_age_s':.5,'max_plan_age_s':.8,'max_arm_tracking_error_rad':.25}
        latch=CommandLatch(.12,gripper_mode='position')
        latch.update(self.command,[2.,4.],time.monotonic())
        q=np.zeros(29);q[15:]=.1
        sent=[]
        robot=types.SimpleNamespace(snapshot=lambda:(q,None,time.monotonic()),
            snapshot_grippers=lambda:(np.array([2.,4.]),time.monotonic()),
            set_command=lambda cmd,grip:sent.append((cmd,np.array(grip))))
        camera=types.SimpleNamespace(snapshot=lambda age:({},time.monotonic()))
        def predict(obs):time.sleep(delay);return 'prediction'
        return c,latch,robot,camera,types.SimpleNamespace(predict=predict),sent

    def test_inference_longer_than_heartbeat_renews_only_validated_hold(self):
        c,latch,robot,camera,policy,sent=self.inference_fixture()
        result=predict_with_hold(policy,{},time.monotonic(),robot,camera,c,self.command,[2,4],latch)
        self.assertEqual(result,'prediction');self.assertGreater(len(sent),3)
        self.assertIsNone(latch.fault)
        for cmd,grip in sent:
            self.assertEqual(cmd['arm_q'],self.command['arm_q'])
            self.assertEqual(cmd['arm_tau'],self.command['arm_tau'])
            self.assertEqual(cmd['base_cmd'][:3],[0,0,0])
            np.testing.assert_equal(grip,[2,4])

    def test_expired_main_loop_is_not_revived_by_hold(self):
        c,latch,robot,camera,policy,sent=self.inference_fixture(delay=0)
        latch.updated_at=time.monotonic()-1
        with self.assertRaisesRegex(RuntimeError,'heartbeat expired'):
            predict_with_hold(policy,{},time.monotonic(),robot,camera,c,self.command,[2,4],latch)
        self.assertFalse(sent)

    def test_hung_inference_cannot_extend_plan_lifetime(self):
        c,latch,robot,camera,policy,sent=self.inference_fixture()
        c['max_plan_age_s']=.15
        with self.assertRaisesRegex(RuntimeError,'reply already expired'):
            predict_with_hold(policy,{},time.monotonic(),robot,camera,c,self.command,[2,4],latch)
        self.assertLess(len(sent),5)

    def test_unhealthy_feedback_or_tracking_stops_hold(self):
        for failure in ('state','camera','tracking'):
            c,latch,robot,camera,policy,sent=self.inference_fixture(delay=0)
            def stale(*args):raise RuntimeError('feedback failure')
            if failure=='state':robot.snapshot=stale
            elif failure=='camera':camera.snapshot=stale
            else:robot.snapshot=lambda:(np.ones(29),None,time.monotonic())
            with self.assertRaises((RuntimeError,ValueError)):
                predict_with_hold(policy,{},time.monotonic(),robot,camera,c,self.command,[2,4],latch)
            self.assertFalse(sent)

    def test_missed_slot_never_publishes_or_renews_heartbeat(self):
        c,latch,robot,camera,policy,sent=self.inference_fixture(delay=0)
        before=latch.updated_at
        self.assertFalse(submit_on_schedule(robot,self.command,[2,4],latch,.08,1/15))
        self.assertEqual(latch.updated_at,before)
        self.assertFalse(sent)
        self.assertTrue(submit_on_schedule(robot,self.command,[2,4],latch,0,1/15))
        self.assertEqual(len(sent),1)

    def test_plan_expiry_during_ik_discards_target_without_renewal(self):
        c,latch,robot,camera,policy,sent=self.inference_fixture(delay=0)
        before=latch.updated_at
        # Even an on-schedule step must not publish an expired plan.
        with patch('g1_deploy.wla_client.time.monotonic',return_value=11):
            self.assertFalse(submit_on_schedule(robot,self.command,[2,4],latch,0,1/15,10,1))
        self.assertFalse(sent)
        self.assertEqual(latch.updated_at,before)
        # A newly observed plan can resume through the normal submission gate.
        self.assertTrue(submit_on_schedule(robot,self.command,[2,4],latch,0,1/15,time.monotonic(),1))
        self.assertEqual(len(sent),1)

    def test_partial_chunks_are_not_counted_as_no_progress(self):
        count=0
        # Actual failing run: five, three, four valid commands, then truncated.
        for accepted in [5,3,4]*10:
            count=check_chunk_progress(count,accepted)
            self.assertEqual(count,0)

    def test_three_empty_chunks_stop_and_valid_step_resets_counter(self):
        count=check_chunk_progress(0,0)
        count=check_chunk_progress(count,0)
        self.assertEqual(count,2)
        self.assertEqual(check_chunk_progress(count,1),0)
        with self.assertRaisesRegex(RuntimeError,'no valid control steps'):
            check_chunk_progress(count,0)

    def test_image_channel_order(self):
        images={k:np.full((480,640,3),[10,20,30],np.uint8) for k in ('head','left_wrist','right_wrist')}
        for image in policy_images(images):np.testing.assert_equal(image[0,0],[30,20,10])

    def test_dataset_contract_and_indices(self):
        c=load_config(ROOT/'config/wla-cup.example.json');self.assertEqual(c['ee_offset'],.105)
        self.assertEqual(execution_indices(),[0,2,4,6,8,10])

    def test_camera_retry_requires_a_fresh_valid_snapshot(self):
        calls=[]
        def snapshot(age):
            calls.append(age)
            if len(calls)==1:raise RuntimeError('DDS camera receive-time skew exceeds limit')
            return {'head':'fresh'},123
        with patch('g1_deploy.wla_client.time.sleep'):
            self.assertEqual(camera_snapshot(types.SimpleNamespace(snapshot=snapshot),{'max_camera_age_s':.5}),({'head':'fresh'},123))
        self.assertEqual(calls,[.5,.5])

    def test_camera_retry_does_not_accept_expired_or_corrupt_images(self):
        def snapshot(age):raise RuntimeError('DDS camera frames missing or stale')
        with patch('g1_deploy.wla_client.time.monotonic',side_effect=[0,1]):
            with self.assertRaisesRegex(RuntimeError,'stale'):
                camera_snapshot(types.SimpleNamespace(snapshot=snapshot),{'max_camera_age_s':.5})
        def bad(age):raise RuntimeError('DDS camera decode: invalid JPEG')
        with self.assertRaisesRegex(RuntimeError,'invalid JPEG'):
            camera_snapshot(types.SimpleNamespace(snapshot=bad),{'max_camera_age_s':.5})

if __name__=='__main__':unittest.main()
