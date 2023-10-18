import os
import argparse
import random
import json
import time
from pathlib import Path
from joint_format import *
from smplx2openpose import smplx2openpose

def main():
    start = time.time()
    
    default_destinationPath = os.path.join(os.path.curdir,'Images','thuman_multiview_renders')
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-objp', '--obj-models-path',
                        type=str,
                        help='Path containing Thuman2.0 *.objs'
                        )
    parser.add_argument('-exrp','--exr-images-path',
                        type=str,
                        help='Path to *.exr images'
                        )
    parser.add_argument('smplx', '--smplx-model-path',
                        type=str,
                        help='Path to SMPL-X Model *.pkl'
                        )
    parser.add_argument('-fpp', '--smplx-fitting-params-path',
                        type=str,
                        help='Path to SMPL-X fitting parameters corresponding to the *.obj models'
                        )
    parser.add_argument('-dp', '--destination-path',
                        type=str,
                        default=default_destinationPath,
                        help='Destination directory to save rendered images'
                        )
    
    args = parser.parse_args()
    
    objs_dir = Path(args.obj_models_path)
    exrs_dir = Path(args.exr_images_path)
    smplx_model_dir = args.smplx_model_path
    fitting_params_dir = args.smplx_fitting_params_path
    dest_path = args.destination_path

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    glob_objs = objs_dir.rglob("*.obj")
    glob_exrs = exrs_dir.rglob("*.exr")

    objs_relative_paths = []
    exrs_relative_paths = []

    for obj_path in glob_objs:
        objs_relative_paths.append(obj_path.relative_to(objs_dir).as_posix())
    for exr_path in glob_exrs:
        exrs_relative_paths.append(exr_path.relative_to(exrs_dir).as_posix())

    random.shuffle(exrs_relative_paths)
    
    smplx_param_name = 'smplx_param.pkl'
    obj_exr_pathPair = dict(zip(objs_relative_paths, exrs_relative_paths))

    for obj_relative_path, exr_relative_path in obj_exr_pathPair.items():
        obj_path = os.path.join(objs_dir, obj_relative_path)
        exr_path = os.path.join(exrs_dir, exr_relative_path)
        fitting_param_path_name = os.path.join(fitting_params_dir, os.path.split(obj_relative_path)[0], smplx_param_name)

        # add openpose rendering component here
        converted_pose_model = smplx2openpose(smplx_model_dir, fitting_param_path_name)
        converted_pose_model.openpose_renderer(destination_path='')

        # add mesh+texture+background rendering component here
        converted_pose_model.mesh_renderer()

        # png_out_model, mesh_center, scale_center = multiview_rendering.create(smpl_path, exr_path)

        png_out_model.write(os.path.join(dest_path, f"{TEST_NAME}.png"))
        # png_out_exr.write(os.path.join(dest_path, f"{TEST_NAME}_hdri.png"))
    
    end = time.time()
    
    print(f"duration: {end-start}")
    
main()