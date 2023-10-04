import open3d as o3d
import numpy as np

def remove_wrists_from_smpl():
    pcd = o3d.io.read_point_cloud('Thuman2_data/Thuman2_SMPL_fittings/THuman2.0_Release/0525/SMPL_joints.ply')
    o3d.visualization.draw_geometries([pcd])
    pcd_np = np.asarray(pcd.points)
    pcd_removed = np.delete(pcd_np,[20,21],axis=0)
    