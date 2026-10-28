"""Compare deployed WLA geometry with real dataset poses, without DDS."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_deploy.wla_client import Kinematics,load_config

p=argparse.ArgumentParser();p.add_argument('--config',default='config/wla-cup.example.json')
p.add_argument('--parquet',default='/tmp/stack-block-file-000.parquet')
p.add_argument('--output',required=True);args=p.parse_args()
c=load_config(args.config);kin=Kinematics(c)
columns=['observation.state.'+s for s in ('left_arm','right_arm','lower_body','left_ee_pose_gripper_base','right_ee_pose_gripper_base','recomputed_ee_valid')]+['action.left_ee_pose_gripper_base','action.right_ee_pose_gripper_base','episode_index','frame_index']
d=pq.read_table(args.parquet,columns=columns).to_pydict();valid=np.flatnonzero(np.array(d['observation.state.recomputed_ee_valid']).reshape(-1))
indices=valid[np.linspace(0,len(valid)-1,200,dtype=int)]
records=[]
for i in indices:
    q=np.r_[d['observation.state.lower_body'][i],d['observation.state.left_arm'][i],d['observation.state.right_arm'][i]]
    fk=kin.fk(q);targets=[]
    row={'episode':int(d['episode_index'][i]),'frame':int(d['frame_index'][i]),'fk_error_mm':[],'fk_rotation_error_deg':[]}
    for s,a in zip(('left','right'),fk):
        label=np.array(d[f'observation.state.{s}_ee_pose_gripper_base'][i])
        row['fk_error_mm'].append(float(np.linalg.norm(a[:3,3]-label[:3])*1000))
        row['fk_rotation_error_deg'].append(float(np.rad2deg((R.from_matrix(a[:3,:3])*R.from_euler('xyz',label[3:]).inv()).magnitude())))
        action=np.array(d[f'action.{s}_ee_pose_gripper_base'][i]);t=np.eye(4);t[:3,3]=action[:3];t[:3,:3]=R.from_euler('xyz',action[3:]).as_matrix();targets.append(t)
    try:
        cmd=kin.command(targets,q,q[15:],np.r_[np.zeros(3),q[[12,14,13]],c['hold_height']],c)
        row.update(kin.last_diagnostic)
    except Exception as exc:row['error']=str(exc)
    records.append(row)
good=[r for r in records if 'error' not in r]
summary={'samples':len(records),'episodes':len(set(r['episode'] for r in records)),'command_checks_passed':len(good),
         'fk_max_position_mm':max(max(r['fk_error_mm']) for r in records),
         'fk_max_rotation_deg':max(max(r['fk_rotation_error_deg']) for r in records),
         'raw_ik_max_position_mm':max(max(r['raw_position_error_mm']) for r in good),
         'raw_ik_max_rotation_deg':max(max(r['raw_rotation_error_deg']) for r in good),
         'max_abs_gravity_nm':max(r['max_abs_tau_nm'] for r in good)}
Path(args.output).write_text(json.dumps({'summary':summary,'rows':records},indent=2));print(json.dumps(summary,indent=2))
if len(good)!=len(records) or summary['fk_max_position_mm']>0.001:sys.exit(1)
