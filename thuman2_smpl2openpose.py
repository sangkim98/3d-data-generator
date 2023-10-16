import os
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import mitsuba as mi
from joint_format import *
import multiview_rendering
from smplx2openpose import smplx2openpose

TEST_NAME = 'test4'

def remove_wrists_from_smpl(path):
    smpl_path = Path(path).glob('*.ply')
    
    for path in smpl_path:
        smpl_path = path
    
    pcd = o3d.io.read_point_cloud(smpl_path.as_posix())
    pcd_np = np.asarray(pcd.points)
    pcd_removed = np.delete(pcd_np,[20,21],axis=0)
    
    return pcd_removed

def main():
    default_destinationPath = os.path.join(os.path.curdir,'Images','openpose_results')
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-sp', '--smplPath',
                        type=str,
                        help='Path to *.ply dataset'
                        )
    parser.add_argument('-exrp','--exrPath',
                        type=str,
                        help='Path to *.exr image'
                        )
    parser.add_argument('-dp', '--destinationPath',
                        type=str,
                        default=default_destinationPath,
                        help='Destination directory to save OpenPose images'
                        )
    
    args = parser.parse_args()
    
    smpl_path = args.smplPath
    exr_path = args.exrPath
    dest_path = args.destinationPath

    if not os.path.exists(args.destinationPath):
        os.makedirs(args.destinationPath)

    openpose_params = smplx2openpose('/home/notingcode/Projects/3d_visualize/models/smplx',
                                     '/home/notingcode/Projects/3d_visualize/Thuman2_data/Thuman2_SMPLX/0026/smplx_param.pkl'
                                     )
    
    openpose_joints = openpose_params.joints
    scale = openpose_params.scale

    png_out_model, png_out_exr, mesh_center, scale_center = multiview_rendering.create_hdri(smpl_path, exr_path)

    png_out_model.write(os.path.join(dest_path, f"{TEST_NAME}.png"))
    png_out_exr.write(os.path.join(dest_path, f"{TEST_NAME}_hdri.png"))
    
    point_colors = []
    joint_pair_idxs = []
    for color_rgb in OPENPOSE_18JOINT_COLOR.values():
        point_colors.append(color_rgb)
    for joint_pair in OPENPOSE_18JOINT_PAIRS:
        joint_pair_idxs.append([OPENPOSE_18JOINT_MAP[joint_pair[0]],OPENPOSE_18JOINT_MAP[joint_pair[1]]])
        
    point_colors = np.asarray(point_colors)
    point_colors = np.divide(point_colors,255)
    joint_pair_idxs = np.asarray(joint_pair_idxs)
    joint_pair_colors = np.asarray(OPENPOSE_18JOINT_PAIRS_COLOR)
    joint_pair_colors = np.divide(joint_pair_colors,255)
    
    pcd = o3d.geometry.PointCloud()
    lineSet = o3d.geometry.LineSet()
    
    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_point = o3d.visualization.rendering.MaterialRecord()

    mat_line.shader = "unlitLine"
    mat_line.line_width = 15
    mat_point.point_size = 15
    
    renderer = o3d.visualization.rendering.OffscreenRenderer(1024,1024)
    renderer.scene.set_background(np.array([0,0,0,0]))
    pcd.points = o3d.utility.Vector3dVector(openpose_joints)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    pcd.scale(scale, center=scale_center)
    pcd.scale(1.3, center=scale_center)
    lineSet.points = pcd.points
    lineSet.lines = o3d.utility.Vector2iVector(joint_pair_idxs)
    lineSet.colors = o3d.utility.Vector3dVector(joint_pair_colors)
    renderer.scene.add_geometry("pcd", pcd, mat_point)
    renderer.scene.add_geometry("lineset", lineSet, mat_line)
    
    camera_position = {'front': [0,0.6+mesh_center[1],1.5], 'back': [0,0.6+mesh_center[1],-1.5], 'left': [1.5,0.6+mesh_center[1],0], 'right': [-1.5,0.6+mesh_center[1],0]}
    
    for angle, value in camera_position.items():
        renderer.setup_camera(60,
                              mesh_center,
                              np.array(value),
                              np.array([0,1,0])
                              )
        img_o3d = renderer.render_to_image()
        o3d.io.write_image(os.path.join(args.destinationPath, f"{TEST_NAME}_{angle}.png"), img_o3d)
    
main()