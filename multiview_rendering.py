import os
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import mitsuba as mi
from mdm2openpose import mdm2openpose
from joint_format import *

# def read_camera_params(filepath: str):
#     if os.path.exists(filepath):
#         camera_params = json.load(open(filepath))
#     else:
#         return None
    
#     mi_camera_params = dict()
    
#     camera_type = camera_params['type']
    
#     if camera_type == 'batch':
#         cameras_params = camera_params['cameras']
        
#         mi_camera_params['type'] = camera_type
        
#         for camera_name, paramters in camera_params.items():
#             mi_camera_params[camera_name] = {'type' : paramters['type'],
#                                              'fov' : paramters['fov'],
#                                              'to_world' : mi.ScalarTransform4f.look_at(origin = )}
#     elif:
#         pass

def render_mesh(mesh, mesh_center, exr_path: str):
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'light': {
            # NOTE: For better results comment out the constant emitter above
            # and uncomment out the lines below changing the filename to an HDRI
            # envmap you have.
            'type': 'envmap',
            'filename': exr_path,
            'scale': 1.0,
        },
        'sensor': {
            'type': 'batch',
            'sensor1': {
                'type': 'perspective',
                'fov': 60,
                'to_world': mi.ScalarTransform4f.look_at(origin=[0,0.6+mesh_center[1],1.5],
                                                         target=mesh_center,
                                                         up=[0,1,0]
                                                         )
            },
            'sensor2': {
                'type': 'perspective',
                'fov': 60,
                'to_world': mi.ScalarTransform4f.look_at(origin=[0,0.6+mesh_center[1],-1.5],
                                                         target=mesh_center,
                                                         up=[0,1,0]
                                                         )
            },
            'sensor3': {
                'type': 'perspective',
                'fov': 60,
                'to_world': mi.ScalarTransform4f.look_at(origin=[1.5,0.6+mesh_center[1],0],
                                                         target=mesh_center,
                                                         up=[0,1,0]
                                                         )
            },
            'sensor4': {
                'type': 'perspective',
                'fov': 60,
                'to_world': mi.ScalarTransform4f.look_at(origin=[-1.5,0.6+mesh_center[1],0],
                                                         target=mesh_center,
                                                         up=[0,1,0]
                                                         )
            },
            'thefilm': {
                'type': 'hdrfilm',
                'width': 1024*4,
                'height': 1024,
            },
            'thesampler': {
                'type': 'multijitter',
                'sample_count': 64,
            },
        },
        'themesh': mesh,
    })

    img = mi.render(scene, spp=256)
    return img

def create_hdri(model_path, exr_path):
    mi.set_variant('cuda_ad_rgb')
    
    obj_path = Path(model_path).glob('*.obj')
    texture_path = Path(model_path).glob('*.jpeg')
    
    for path in obj_path:
        obj_path = path
    for path in texture_path:
        texture_path = path
    
    mesh = o3d.io.read_triangle_mesh(obj_path.as_posix(), True)
    mesh.textures = [o3d.io.read_image(texture_path.as_posix())]

    scale_center = mesh.get_center()

    mesh.scale(1.3, center=scale_center)    
    
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh_center = tensor_mesh.get_axis_aligned_bounding_box().get_center()
    
    mi_mesh = tensor_mesh.to_mitsuba('thuman')
    img = render_mesh(mi_mesh, mesh_center.numpy(), exr_path)
    png_out_model = mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt16, True)
    
    exr = mi.Bitmap(exr_path)
    png_out_exr = exr.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt16, True)
    
    return png_out_model, png_out_exr, mesh_center.numpy(), scale_center