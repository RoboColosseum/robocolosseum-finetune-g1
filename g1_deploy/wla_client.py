"""WLA short-horizon policy using g01's shared WBC command transport.

Dataset-compatible measured-waist FK/IK is separate from the shared reference
controller's arm filtering, gravity compensation, and held-command publishing.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
from .core import vector, slew, CommandLatch, hold_command
from .wbc_client import CameraReader, Kinematics as ReferenceKinematics, ArmTrackingError

LOG = logging.getLogger(__name__)

def load_config(path):
    c=json.loads(Path(path).read_text())
    if (c['model_action_hz'],c['action_hz'],c['execute_steps']) != (30,15,6):
        raise ValueError('This deployment is validated for 30 Hz -> 15 Hz, six steps')
    if c.get('gripper_command_mode')!='position' or not c.get('subscribe_gripper_state'):
        raise ValueError('WLA requires absolute Dex1 positions and measured Dex1 state')
    for key in ('command_heartbeat_enabled','robot_state_expiry_enabled'):
        if c.get(key) is not True:raise ValueError(f'{key} must be enabled')
    for key in ('command_timeout_s','request_timeout_s','startup_timeout_s','max_state_age_s','max_camera_age_s',
                'max_plan_age_s','arm_speed_rad_s','max_arm_tracking_error_rad','max_arm_feedforward_nm',
                'max_ik_position_error_m','max_ik_rotation_error_rad','max_ee_translation_m','max_ee_rotation_rad','gripper_speed_per_s'):
        if not np.isfinite(c[key]) or c[key]<=0:raise ValueError(f'Invalid {key}')
    obs_age=c.get('observation_camera_max_age_s',.1)
    if not np.isfinite(obs_age) or not 0<obs_age<=c['max_camera_age_s'] or obs_age>=c['max_plan_age_s']:
        raise ValueError('Invalid observation_camera_max_age_s')
    if c['ee_offset']!=.105 or c['left_frame']!='left_wrist_yaw_link' or c['right_frame']!='right_wrist_yaw_link':
        raise ValueError('Dataset tool frame must remain wrist-local +X 0.105 m')
    if not np.isfinite([c['gripper_min'],c['gripper_max'],c['hold_height']]).all() or not c['gripper_min']<c['gripper_max']:
        raise ValueError('Invalid gripper limits or height')
    if c['camera'].get('head_mode','left_eye')!='left_eye':raise ValueError('WLA needs head-left image')
    return c

def policy_images(images):
    """Shared CameraReader returns RGB; the existing WLA server expects BGR."""
    result=[]
    for name in ('head','left_wrist','right_wrist'):
        a=np.asarray(images[name])
        if a.shape!=(480,640,3) or a.dtype!=np.uint8:raise ValueError(f'Invalid {name} image')
        result.append(np.ascontiguousarray(a[:,:,::-1]))
    return tuple(result)

def execution_indices():
    return [0,2,4,6,8,10]

def camera_snapshot(camera,c,max_age=None):
    """Wait briefly for all three fresh frames after a blocking policy call.

    This does not widen freshness/skew limits. The caller checks plan age and
    schedule again before submitting any command; transport heartbeat stays on.
    """
    deadline=time.monotonic()+.3
    while True:
        try:return camera.snapshot(c['max_camera_age_s'] if max_age is None else max_age)
        except RuntimeError as exc:
            if (time.monotonic()>=deadline or
                not any(s in str(exc) for s in ('missing or stale','receive-time skew'))):raise
            time.sleep(.01)

def submit_command(robot, command, grip, dry_latch=None):
    """Dry-run exercises the same heartbeat without creating DDS publishers."""
    now = time.monotonic()
    if dry_latch is not None:
        dry_latch.sample(now, True)
        if dry_latch.fault:
            raise RuntimeError(dry_latch.fault)
        dry_latch.update(command, grip, now)
    robot.set_command(command, grip)


def submit_on_schedule(robot, command, grip, dry_latch, late_s, period,
                       captured=None, max_plan_age=None):
    # A missed slot must not execute an old target or cause a catch-up burst.
    if captured is not None and time.monotonic()-captured >= max_plan_age:
        return False
    if late_s > period:
        return False
    submit_command(robot, command, grip, dry_latch)
    return True


def check_chunk_progress(previous_empty_chunks, accepted_steps):
    """Partial valid chunks are progress; repeated empty chunks stop retries."""
    empty_chunks = 0 if accepted_steps else previous_empty_chunks + 1
    if empty_chunks >= 3:
        raise RuntimeError('Three consecutive chunks produced no valid control steps; stopping')
    return empty_chunks


def predict_with_hold(policy, obs, captured, robot, camera, c,
                      previous_command=None, previous_grip=None, dry_latch=None):
    """Bounded inference wait; only the healthy main loop renews the hold.

    The worker cannot publish or renew heartbeat. Plan age remains bounded even
    if a server hangs. Camera/state/remote-stop/tracking failures stop renewal.
    """
    done = threading.Event()
    result = {}
    def predict():
        try:
            result['actions'] = policy.predict(obs)
        except BaseException as exc:
            result['error'] = exc
        finally:
            done.set()
    worker = threading.Thread(target=predict, daemon=True, name='wla-inference')
    worker.start()
    while True:
        if time.monotonic() - captured >= c['max_plan_age_s']:
            raise RuntimeError('Policy reply already expired')
        if previous_command is not None:
            camera_snapshot(camera, c)
            q, _, _ = robot.snapshot()
            robot.snapshot_grippers()
            if np.max(np.abs(np.asarray(previous_command['arm_q']) - q[15:])) > c['max_arm_tracking_error_rad']:
                raise ArmTrackingError('Inference hold exceeds measured joint tracking limit')
            if time.monotonic() - captured >= c['max_plan_age_s']:
                raise RuntimeError('Policy reply already expired')
            submit_command(robot, hold_command(previous_command), previous_grip, dry_latch)
        if done.is_set():
            if 'error' in result:
                raise result['error']
            return result['actions']
        done.wait(.05)

class Kinematics:
    def __init__(self,c):
        sys.path.insert(0,str(Path(c['wla_repo']).expanduser()))
        from model_client.wbc_kinematics import WBCKinematics
        from model_client.g1_core import BODY_JOINTS
        self.dataset=WBCKinematics(c['urdf'],c['left_frame'],c['right_frame'],c['ee_offset'])
        self.reference=ReferenceKinematics(c['wbc_repo'])
        model=self.reference.ik.reduced_robot.model
        if list(model.names)[1:]!=list(BODY_JOINTS[15:]):
            raise ValueError('Reference gravity arm joint order differs from dataset SDK order')
        self.last_diagnostic=None

    def fk(self,q):return self.dataset.fk(q)

    def warmup(self,q):self.dataset.warmup(q)

    def command(self,targets,q,previous,base,c):
        raw=vector(self.dataset.ik(targets,q),14,'dataset IK')
        solved=np.asarray(q).copy();solved[15:]=raw
        actual=self.dataset.fk(solved)
        positions=[float(np.linalg.norm(a[:3,3]-t[:3,3])) for a,t in zip(actual,targets)]
        angles=[float(Rotation.from_matrix(a[:3,:3]@t[:3,:3].T).magnitude()) for a,t in zip(actual,targets)]
        if max(positions)>c['max_ik_position_error_m'] or max(angles)>c['max_ik_rotation_error_rad']:
            raise ValueError(f'Dataset IK residual too large: {positions} m / {angles} rad')
        # Reuse the g01 reference filter, but never its 5 cm/zero-waist FK or IK.
        self.reference.ik.smooth_filter.add_data(raw)
        filtered=vector(self.reference.ik.smooth_filter.filtered_data,14,'filtered IK')
        arm=slew(previous,filtered,c['arm_speed_rad_s'],1/c['action_hz'])
        if np.max(np.abs(arm-q[15:]))>c['max_arm_tracking_error_rad']:
            raise ArmTrackingError('WLA command exceeds measured joint tracking limit')
        model=self.reference.ik.reduced_robot.model
        lo=np.maximum(self.dataset.limits[0],model.lowerPositionLimit)
        hi=np.minimum(self.dataset.limits[1],model.upperPositionLimit)
        if np.any(arm<lo) or np.any(arm>hi):raise ValueError('Arm target outside model joint limits')
        import pinocchio as pin
        # Same model and RNEA-at-slew-limited-target calculation as wbc_client.
        tau=vector(pin.rnea(model,self.reference.ik.reduced_robot.data,arm,np.zeros(14),np.zeros(14)),14,'gravity torque')
        if np.max(np.abs(tau))>c['max_arm_feedforward_nm']:raise ValueError('Gravity torque too large')
        self.last_diagnostic={'raw_position_error_mm':(np.array(positions)*1000).tolist(),
                              'raw_rotation_error_deg':np.rad2deg(angles).tolist(),
                              'raw_arm':raw.tolist(),'command_arm':arm.tolist(),'max_abs_tau_nm':float(np.max(np.abs(tau)))}
        return {'arm_q':arm.tolist(),'arm_tau':tau.tolist(),'base_cmd':vector(base,7,'held base').tolist()}

def run(c,mode,cycles,save):
    import cv2
    cv2.setNumThreads(1)  # Avoid CPU oversubscription against DDS/JPEG and IK threads.
    sys.path.insert(0,str(Path(c['wla_repo']).expanduser()))
    from model_client.g1_core import PolicyClient, build_obs, pose_matrix, resample_actions
    from .transport import RobotTransport
    kin=Kinematics(c)
    if mode=='check':
        LOG.info('Dataset IK and shared g01 gravity model loaded; no DDS or policy connection')
        return
    robot=camera=policy=None
    records=[]
    last_observation = None
    dry_latch = CommandLatch(c['command_timeout_s'], gripper_mode='position') if mode=='dry-run' else None
    previous_command = previous_grip = None
    try:
        robot=RobotTransport(c,execute=mode=='execute')
        camera=CameraReader(c['camera']);robot.wait_ready()
        deadline=time.monotonic()+c['startup_timeout_s']
        while True:
            try:camera.snapshot(c['max_camera_age_s']);break
            except RuntimeError:
                if time.monotonic()>deadline:raise
                time.sleep(.05)
        q,_,_=robot.snapshot();grips,_=robot.snapshot_grippers()
        kin.warmup(q)
        policy=PolicyClient(c['server'],c['unnorm_key'],c['request_timeout_s'])
        if policy.horizon!=30:raise ValueError('Expected WLA 30-step horizon')
        images,_=camera_snapshot(camera,c)
        policy.predict(build_obs(q,grips,policy_images(images),kin.fk(q),c['task'],c['unnorm_key']))
        LOG.info('Warmup discarded. mode=%s; shared g01 transport; model indices=%s',mode,execution_indices())
        q,_,_=robot.snapshot();last_q=q[15:].copy()
        base=np.r_[np.zeros(3),q[[12,14,13]],c['hold_height']]
        cycle=0
        consecutive_empty_chunks=0
        while cycles==0 or cycle<cycles:
            q,_,state_time=robot.snapshot();grips,grip_time=robot.snapshot_grippers()
            images,image_time=camera_snapshot(camera,c,max_age=c.get('observation_camera_max_age_s',.1))
            q,_,state_time=robot.snapshot();grips,grip_time=robot.snapshot_grippers()
            captured=min(state_time,grip_time,image_time)
            anchors=kin.fk(q);obs=build_obs(q,grips,policy_images(images),anchors,c['task'],c['unnorm_key'])
            observed_q=q.copy()
            last_observation = (obs, observed_q)
            start=time.monotonic()
            actions=predict_with_hold(policy,obs,captured,robot,camera,c,previous_command,previous_grip,dry_latch)
            infer_s=time.monotonic()-start
            plan=resample_actions(actions,c['model_action_hz'],c['action_hz'])
            if time.monotonic()-captured>=c['max_plan_age_s']:raise RuntimeError('Policy reply already expired')
            record={'cycle':cycle+1,'inference_s':infer_s,'mode':mode,'steps':[]}
            records.append(record)
            started=time.monotonic()
            for step in range(c['execute_steps']):
                step_started=time.monotonic()
                if step_started-captured>=c['max_plan_age_s']:
                    record['rejected_step']={'step':step+1,'reason':'plan_expired_before_ik',
                                             'age_s':step_started-captured}
                    LOG.warning('Discarding expired chunk and replanning: %s',record['rejected_step'])
                    break
                # The plan already has validated images. Check cameras once,
                # after IK and immediately before publishing; do not wait for
                # synchronized images twice for the same command.
                q,_,_=robot.snapshot();robot.snapshot_grippers()
                targets=[pose_matrix(plan[s+'_ee_rpy'][step]) for s in ('left','right')]
                for target,anchor in zip(targets,anchors):
                    if (np.linalg.norm(target[:3,3]-anchor[:3,3])>c['max_ee_translation_m'] or
                        Rotation.from_matrix(target[:3,:3]@anchor[:3,:3].T).magnitude()>c['max_ee_rotation_rad']):
                        raise ValueError('EE target too far from current observation')
                t=time.monotonic()
                command=kin.command(targets,q,last_q if mode=='execute' else q[15:],base,c)
                ik_s=time.monotonic()-t
                grip=np.clip([plan['left_gripper'][step,0],plan['right_gripper'][step,0]],c['gripper_min'],c['gripper_max'])
                camera_started=time.monotonic()
                camera_snapshot(camera,c)
                camera_s=time.monotonic()-camera_started
                fresh_q,_,_=robot.snapshot();robot.snapshot_grippers()
                if np.max(np.abs(np.array(command['arm_q'])-fresh_q[15:]))>c['max_arm_tracking_error_rad']:
                    raise ArmTrackingError('Post-IK feedback tracking limit')
                target_time=started+(step+1)/c['action_hz']
                late_s=max(0.,time.monotonic()-target_time)
                timing={'step':step+1,'ik_s':ik_s,'camera_check_s':camera_s,
                        'step_compute_s':time.monotonic()-step_started,'schedule_late_s':late_s}
                if not submit_on_schedule(robot,command,grip,dry_latch,late_s,1/c['action_hz'],
                                          captured,c['max_plan_age_s']):
                    age_s=time.monotonic()-captured
                    record['rejected_step']={**timing,'age_s':age_s,
                        'reason':'plan_expired_after_ik' if age_s>=c['max_plan_age_s'] else 'schedule_missed'}
                    LOG.warning('Discarding remaining chunk and replanning: %s',record['rejected_step'])
                    break
                previous_command,previous_grip=command,grip.copy()
                last_q=np.array(command['arm_q'])
                entry={**timing,'model_index':execution_indices()[step],
                       'age_s':time.monotonic()-captured,'command':command,'gripper':grip.tolist(),**kin.last_diagnostic}
                record['steps'].append(entry)
                LOG.info('cycle=%d step=%d raw IK mm=%s age=%.3fs IK=%.1fms camera=%.1fms late=%.1fms mode=%s',
                         cycle+1,step+1,entry['raw_position_error_mm'],entry['age_s'],
                         ik_s*1000,camera_s*1000,late_s*1000,mode)
                time.sleep(max(0,target_time-time.monotonic()))
            # Retain last accepted arm_q/arm_tau; never overwrite with measured q
            # while waiting for the next inference. Shared transport publishes it.
            consecutive_empty_chunks=check_chunk_progress(consecutive_empty_chunks,len(record['steps']))
            cycle+=1
    except BaseException:
        LOG.exception('WLA control loop stopped before cleanup')
        raise
    finally:
        # Shared transport close delivers last-command hold, never measured-pose reset.
        watchdog=threading.Timer(8,lambda:os._exit(2));watchdog.daemon=True;watchdog.start()
        try:
            if robot:robot.close()
            # Save only after stopping command transport, but before camera/DDS
            # teardown, which can block and trigger the cleanup watchdog.
            if save:
                folder=Path(save);folder.mkdir(parents=True,exist_ok=True)
                (folder/'cycles.json').write_text(json.dumps(records,indent=2))
                if last_observation is not None:
                    obs,observed_q=last_observation
                    np.savez_compressed(folder/'last-observation.npz',**obs,robot_q29=observed_q)
            if camera:camera.close()
            if policy:policy.close()
        finally:
            watchdog.cancel()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/wla-cup.example.json')
    p.add_argument('--mode',choices=('check','dry-run','execute'),default='dry-run')
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--instruction')
    p.add_argument('--save')
    a=p.parse_args()
    if a.cycles<0:p.error('cycles must be >=0')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    c=load_config(a.config)
    if a.instruction:c['task']=a.instruction
    run(c,a.mode,a.cycles,a.save)

if __name__=='__main__':main()
