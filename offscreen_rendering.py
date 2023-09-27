import open3d as o3d
import numpy as np
import cv2
import os
from pathlib import Path
from mdm_to_openpose import mdm2openpose
from joint_format import *

def main():
    mdm_path = Path("~/Projects/camera_pose/save_result/results_alice_v2.npy")

    test = mdm2openpose(mdm_path)

    camera_position = {'left': -0.8,'center': 0,'right': 0.8}
    camera_zoom = {'left': 1.1,'center': 1.3,'right': 1.1}

    #######
    point_colors = []
    joint_pair_idxs = []
    for color_rgb in OPENPOSE_JOINT_COLOR.values():
        point_colors.append(color_rgb)
    for joint_pair in OPENPOSE_JOINT_PAIRS:
        joint_pair_idxs.append([OPENPOSE_JOINT_MAP[joint_pair[0]],OPENPOSE_JOINT_MAP[joint_pair[1]]])
        
    point_colors = np.asarray(point_colors)
    point_colors = np.divide(point_colors,255)
    joint_pair_idxs = np.asarray(joint_pair_idxs)
    joint_pair_colors = np.asarray(OPENPOSE_JOINT_PAIRS_COLOR)
    joint_pair_colors = np.divide(joint_pair_colors,255)
    ########
    
    pcd = o3d.geometry.PointCloud()
    lineSet = o3d.geometry.LineSet()
    
    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_point = o3d.visualization.rendering.MaterialRecord()

    mat_line.shader = "unlitLine"
    mat_line.line_width = 15
    mat_point.point_size = 15
    
    for action_idx in range(test.openpose_motion.shape[0]):

        entire_motion = test.openpose_motion[action_idx,:,:,:]

        for frame_idx in range(entire_motion.shape[2]//3):
            if frame_idx % 30 == 0:
                renderer = o3d.visualization.rendering.OffscreenRenderer(1024,1024)
                renderer.scene.set_background(np.array([0,0,0,0]))
                pcd.points = o3d.utility.Vector3dVector(entire_motion[:,:,frame_idx])
                pcd.colors = o3d.utility.Vector3dVector(point_colors)
                lineSet.points = o3d.utility.Vector3dVector(entire_motion[:,:,frame_idx])
                lineSet.lines = o3d.utility.Vector2iVector(joint_pair_idxs)
                lineSet.colors = o3d.utility.Vector3dVector(joint_pair_colors)
                center = pcd.get_center()
                renderer.scene.add_geometry("pcd",pcd,mat_point)
                renderer.scene.add_geometry("lineset",lineSet,mat_line)
                
                for angle, value in camera_position.items():
                    renderer.setup_camera(70,
                                          np.array([center[0],center[1]-0.2,center[2]]),
                                          np.array([center[0]+value,center[1]+0.1,center[2]+camera_zoom[angle]]),
                                          np.array([0,1,0])
                                          )
                    img_o3d = renderer.render_to_image()
                    o3d.io.write_image(os.path.join("/home/notingcode/Projects/camera_pose/openpose_images", f"{action_idx}_{frame_idx}_{angle}.png"), img_o3d)
                
                del renderer
    
main()