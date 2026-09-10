"""Quick test: cube at X=0.50 works with IK_BIAS=0 + 1000N?"""
import math, os, sys, time
import cv2
import numpy as np
import pybullet as p
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.configs.config import CAM_WIDTH, CAM_HEIGHT, TABLE_SURFACE_Z, MULTI_OBJECTS_CONFIG
from src.env.sim_env import setup_simulation_multi, init_panda_pose, stabilize_objects, load_single_object
from src.perception.camera import get_camera_image
from src.perception.detector import YoloSegmentor
from src.perception.pose_estimator import PoseEstimator
from src.control.ik_controller import move_to_pose

END_EFF = 8; FINGER_OFFSET = 0.105
EST = PoseEstimator(cy_offset=-54, verbose=False)
FUSION_EYES = [[1.3,0,1.4], [0.5,-1,1.2], [-0.2,-0.5,1.0], [0.8,0.8,1.5]]

# Load with cube at X=0.50
import pybullet_data
client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", [0.4,0,0])
robot_id = p.loadURDF("franka_panda/panda.urdf", [0,0,0.5], useFixedBase=True)
init_panda_pose(robot_id)
obj_id = load_single_object("cube_small.urdf", [0.50, 0.0, 0.66], [0,0,30], 1.0)
stabilize_objects([{"name":"cube","obj_id":obj_id,"pos":[0.50,0.0,0.66]}], 240)

gt = p.getBasePositionAndOrientation(obj_id)[0]
aabb_min, aabb_max = p.getAABB(obj_id)
print(f"Cube GT: ({gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f})")
print(f"AABB Z: [{aabb_min[2]:.3f}, {aabb_max[2]:.3f}] h={aabb_max[2]-aabb_min[2]:.3f}m")

yolo = YoloSegmentor(model_path="models/custom_yolov8n_seg.pt")

# pose estimate
rgb1, depth1, _, _, vm1, _ = get_camera_image()
dets = yolo.detect_all(rgb1)
t = next((d for d in dets if d["cls_id"] == 2), None)
if not t: print("no detect"); exit()
mask = cv2.resize(t["mask"], (CAM_WIDTH, CAM_HEIGHT), cv2.INTER_NEAREST)
mb = (mask > 0.5).astype(np.uint8)
mb = cv2.erode(mb, np.ones((3,3),np.uint8),1)
fs = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.int32)
fs[mb > 0] = 9999
pts1 = EST.extract_view_points(depth1, fs, vm1, 9999)
rough_xy = np.median(pts1[:,:2], axis=0)

best_xy = rough_xy.copy()
all_pts = None
for iteration in range(3):
    ct = [float(best_xy[0]), float(best_xy[1]), 0.65]
    c3, pl, ms = [], [], []
    for eye in FUSION_EYES:
        rgb, d, _, _, vm, _ = get_camera_image(eye=eye, target=ct)
        dets = yolo.detect_all(rgb)
        t = next((d for d in dets if d["cls_id"] == 2), None)
        if not t: continue
        m = cv2.resize(t["mask"], (CAM_WIDTH, CAM_HEIGHT), cv2.INTER_NEAREST)
        mb = (m > 0.5).astype(np.uint8)
        mb = cv2.erode(mb, np.ones((3,3),np.uint8),1)
        px = np.sum(mb)
        if px < 100: continue
        ffs = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.int32)
        ffs[mb > 0] = 9999
        pts = EST.extract_view_points(d, ffs, vm, 9999)
        if pts is None or len(pts) < 30: continue
        c3.append(np.median(pts, axis=0))
        pl.append(pts)
        ms.append(px)
    if len(c3) < 2: break
    ca, msz = np.array(c3), np.array(ms, np.float64)
    n = len(ca)
    if n >= 3:
        loo = np.ones(n)
        for i in range(n):
            o = np.delete(ca,i,axis=0)
            loo[i] = 1.0/max(np.linalg.norm(ca[i]-np.median(o,axis=0)),1e-6)
        loo /= loo.sum(); w = loo * msz
    else: w = msz
    w /= w.sum()
    np2 = np.zeros(3)
    for dim in range(3):
        order = np.argsort(ca[:,dim])
        cw = np.cumsum(w[order])
        idx = min(np.searchsorted(cw, 0.5), n-1)
        np2[dim] = ca[order[idx], dim]
    if np.linalg.norm(np2[:2]-best_xy)*100 < 2.0 and iteration > 0: break
    best_xy = np2[:2]
    all_pts = np.vstack(pl) if len(pl) > 1 else pl[0]

grasp_z = max(float(np.min(all_pts[:,2])), TABLE_SURFACE_Z) + 0.15*max(float(np.percentile(all_pts[:,2],95))-max(float(np.min(all_pts[:,2])), TABLE_SURFACE_Z), 0.005)
ox, oy = best_xy[0], best_xy[1]
gt_xy_err = np.linalg.norm(best_xy - gt[:2])*100
print(f"Est: ({ox:.3f},{oy:.3f}) GT_err={gt_xy_err:.1f}cm grasp_z={grasp_z:.3f}")

# Grasp with IK_BIAS=0 + 1000N
fine_z = grasp_z + FINGER_OFFSET
appr_z = fine_z + 0.03
pre_z = fine_z + 0.20
dq = p.getQuaternionFromEuler([math.pi, 0, 0])

def ctrl(w, s=200, f=350):
    for j in [9,10]:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, w, force=f)
    for _ in range(s): p.stepSimulation()

ctrl(0.04, 100)
move_to_pose(robot_id, [ox,oy,pre_z], dq, END_EFF, 200)
move_to_pose(robot_id, [ox,oy,appr_z], dq, END_EFF, 300)
move_to_pose(robot_id, [ox,oy,fine_z], dq, END_EFF, 800, convergence_threshold=1e-3)

ee = p.getLinkState(robot_id, END_EFF)[0]
ik_err = math.dist([ox,oy,fine_z], ee)*100
ct_before = len(p.getContactPoints(robot_id, obj_id))
print(f"fine: handZ={ee[2]:.3f} fingZ={ee[2]-FINGER_OFFSET:.3f} err={ik_err:.1f}cm contact={ct_before}")

p.changeDynamics(obj_id,-1,lateralFriction=2.0,spinningFriction=1.0)
for j in [9,10]:
    p.changeDynamics(robot_id,j,lateralFriction=5.0,spinningFriction=2.0)

ctrl(0.0, 300, 1000)
for _ in range(150):
    p.setJointMotorControl2(robot_id,9,p.POSITION_CONTROL,0.0,force=1000)
    p.setJointMotorControl2(robot_id,10,p.POSITION_CONTROL,0.0,force=1000)
    p.stepSimulation()
ct_after = len(p.getContactPoints(robot_id, obj_id))
print(f"grasp: contact={ct_after}")

oz0 = p.getBasePositionAndOrientation(obj_id)[0][2]
move_to_pose(robot_id, [ox,oy,fine_z+0.05], dq, END_EFF, 150)
for _ in range(120): p.stepSimulation()
oz1 = p.getBasePositionAndOrientation(obj_id)[0][2]
dz = (oz1-oz0)*100
print(f"verify: Δz={dz:.1f}cm {'✅' if dz>1.0 else '❌'}")

p.disconnect()
