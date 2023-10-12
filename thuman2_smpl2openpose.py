import os
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import mitsuba as mi
from mdm2openpose import mdm2openpose
from joint_format import *

TEST_NAME = 'test2'

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

    png_out_model, png_out_exr, mesh_center, scale_center = create_hdri(smpl_path, exr_path)

    png_out_model.write(os.path.join(dest_path, f"{TEST_NAME}.png"))
    png_out_exr.write(os.path.join(dest_path, f"{TEST_NAME}_hdri.png"))

    new_smpl_np = remove_wrists_from_smpl(smpl_path)

    def convert_mdm2openpose(new_smpl_np):
        def norm_of_3D_plane(v1, v2, v3):
            """_summary_

            Args:
                v1 (_type_): _description_
                v2 (_type_): _description_
                v3 (_type_): _description_

            Returns:
                _type_: _description_
            """
            
            a = v1 - v2
            b = v2 - v3
            
            normal_vector = np.cross(a,b)
            
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
                
            return normal_vector
        
        def create_new_neck():
            """
            
            """
        
            left_shoulder = new_smpl_np[MDM_22JOINT_MAP['LeftShoulder'],:]
            right_shoulder = new_smpl_np[MDM_22JOINT_MAP['RightShoulder'],:]
            spine2 = new_smpl_np[MDM_22JOINT_MAP['Spine2'],:]
            
            new_neck = (left_shoulder+right_shoulder)*0.45 + spine2*0.1
            
            return new_neck
            
        def create_eyes_ears():
            """
            
            """
            
            head = new_smpl_np[MDM_22JOINT_MAP['Head'],:]
            neck = new_smpl_np[MDM_22JOINT_MAP['Neck'],:]
            spine2 = new_smpl_np[MDM_22JOINT_MAP['Spine2'],:]

            normal_vector = norm_of_3D_plane(head, neck, spine2)

            normal_vector_eyes = np.divide(normal_vector,25)
            normal_vector_ears = np.divide(normal_vector,20)

            left_eye = head - normal_vector_eyes
            right_eye = head + normal_vector_eyes

            left_ear = head - normal_vector_ears
            right_ear = head + normal_vector_ears
            
            norm_to_mv_eyes_ears = norm_of_3D_plane(neck, left_eye, right_eye)
            norm_to_mv_eyes = np.divide(norm_to_mv_eyes_ears, 25)
            norm_to_mv_ears = np.divide(norm_to_mv_eyes_ears, 22)
            
            left_eye = left_eye - norm_to_mv_eyes
            right_eye = right_eye - norm_to_mv_eyes
            
            left_ear = left_ear - norm_to_mv_ears
            right_ear = right_ear - norm_to_mv_ears
            
            norm_to_mv_eyes_ears_down = norm_of_3D_plane(head, left_eye, right_eye)
            norm_to_mv_ears_down = np.divide(norm_to_mv_eyes_ears_down, 18)
            
            left_ear = left_ear - norm_to_mv_ears_down
            right_ear = right_ear - norm_to_mv_ears_down
            
            return (right_eye, left_eye, right_ear, left_ear)            
            
        def create_rlHips():                
            """
            
            """
            
            left_upleg = new_smpl_np[MDM_22JOINT_MAP['LeftUpLeg'],:]
            right_upleg = new_smpl_np[MDM_22JOINT_MAP['RightUpLeg'],:]

            left_hip = (left_upleg*1.1+right_upleg*-0.1)                
            right_hip = (left_upleg*-0.1+right_upleg*1.1)
            
            return (right_hip, left_hip)
            
        new_neck = create_new_neck()
        (right_eye, left_eye, right_ear, left_ear) = create_eyes_ears()
        (right_hip, left_hip) = create_rlHips()
        
        data_shape = new_smpl_np.shape
        
        openpose_data_shape = (18, data_shape[1])
        
        openpose_data_edited = np.empty(openpose_data_shape, dtype=new_smpl_np.dtype)
        
        new_joints = {
            'Neck': new_neck,
            'REye': right_eye, 'LEye': left_eye,
            'REar': right_ear, 'LEar': left_ear,
            'RHip': right_hip, 'LHip': left_hip
        }
        
        for mdm_joint, openpose_joint in MDM2OPENPOSE_KEYVAL.items():
            if openpose_joint != REMOVE:
                openpose_data_edited[OPENPOSE_18JOINT_MAP[openpose_joint],:] = new_smpl_np[MDM_22JOINT_MAP[mdm_joint],:]
            
        for new_openpose_joint, data in new_joints.items():
            openpose_data_edited[OPENPOSE_18JOINT_MAP[new_openpose_joint],:] = data
            
                
        return openpose_data_edited

    new_smpl = convert_mdm2openpose(new_smpl_np)

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
    pcd.points = o3d.utility.Vector3dVector(new_smpl)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    pcd.scale(1.3, center=scale_center)
    lineSet.points = pcd.points
    lineSet.lines = o3d.utility.Vector2iVector(joint_pair_idxs)
    lineSet.colors = o3d.utility.Vector3dVector(joint_pair_colors)
    renderer.scene.add_geometry("pcd",pcd,mat_point)
    renderer.scene.add_geometry("lineset",lineSet,mat_line)
    
    camera_position = {'front': [0,0.6+mesh_center[1],1.5], 'back': [0,0.6+mesh_center[1],-1.5], 'left': [1.5,0.6+mesh_center[1],0], 'right': [-1.5,0.6+mesh_center[1],0]}
    
    for angle, value in camera_position.items():
        renderer.setup_camera(60,
                                mesh_center,
                                np.array(value),
                                np.array([0,1,0])
                                )
        img_o3d = renderer.render_to_image()
        o3d.io.write_image(os.path.join(args.destinationPath, f"{angle}.png"), img_o3d)
    
main()