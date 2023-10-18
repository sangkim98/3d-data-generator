import os
import open3d as o3d
import mitsuba as mi
import numpy as np
from joint_format import *

class convert2openpose():
    def __init__(self) -> None:
        self.openpose18_joints = None
        
        point_colors = []
        joint_pair_idxs = []
        for color_rgb in OPENPOSE_18JOINT_COLOR.values():
            point_colors.append(color_rgb)
        for joint_pair in OPENPOSE_18JOINT_PAIRS:
            joint_pair_idxs.append([OPENPOSE_18JOINT_MAP[joint_pair[0]],OPENPOSE_18JOINT_MAP[joint_pair[1]]])
            
        self.openpose_point_colors = np.divide(np.asarray(point_colors), 255)
        self.openpose_joint_pair_idxs = np.asarray(joint_pair_idxs)
        self.openpose_joint_pair_colors = np.divide(np.asarray(OPENPOSE_18JOINT_PAIRS_COLOR), 255)
        
    def convert(self):
        ...
        
    def norm_of_plane(self, v1, v2, v3, axis=1):
        from sklearn.preprocessing import normalize
        
        a = v1 - v2
        b = v2 - v3
        
        normal_vector = np.cross(a,b,axis=axis)
        
        normalize(normal_vector, axis=axis, copy=False)
            
        return normal_vector
    
    def openpose_renderer(self, destination_path: str):
        if self.openpose18_joints is None:
            print("OpenPose18 joints are not stored")
            return None
        
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        
        pcd = o3d.geometry.PointCloud()
        lineSet = o3d.geometry.LineSet()
        
        mat_line = o3d.visualization.rendering.MaterialRecord()
        mat_point = o3d.visualization.rendering.MaterialRecord()

        mat_line.shader = "unlitLine"
        mat_line.line_width = 15
        mat_point.point_size = 15
        
        renderer = o3d.visualization.rendering.OffscreenRenderer(1024,1024)
        renderer.scene.set_background(np.array([0,0,0,0]))
        pcd.points = o3d.utility.Vector3dVector(self.openpose18_joints)
        pcd.colors = o3d.utility.Vector3dVector(self.openpose_point_colors)
        pcd.scale(scale_factor, center=scale_center)
        lineSet.points = pcd.points
        lineSet.lines = o3d.utility.Vector2iVector(self.openpose_joint_pair_idxs)
        lineSet.colors = o3d.utility.Vector3dVector(self.openpose_joint_pair_colors)
        renderer.scene.add_geometry("pcd", pcd, mat_point)
        renderer.scene.add_geometry("lineset", lineSet, mat_line)
        
        camera_position = {'front': [0,0.6+mesh_center[1],1.5],
                           'back': [0,0.6+mesh_center[1],-1.5],
                           'left': [1.5,0.6+mesh_center[1],0],
                           'right': [-1.5,0.6+mesh_center[1],0]
                           }
        
        for angle, value in camera_position.items():
            renderer.setup_camera(60, mesh_center, np.array(value), np.array([0,1,0]))
            img_o3d = renderer.render_to_image()
            o3d.io.write_image(os.path.join(destination_path, f"{}_{angle}.png"), img_o3d)
    
    def mesh_renderer(self):
        def render_mesh(mesh, mesh_center, exr_path: str):
            scene = mi.load_dict({
                'type': 'scene',
                'integrator': {
                    'type': 'path'
                },
                'light': {
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
                        'to_world': mi.ScalarTrans*form4f.look_at(origin=[-1.5,0.6+mesh_center[1],0],
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

        def create(model_path, exr_path):
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
            
            # exr = mi.Bitmap(exr_path)
            # png_out_exr = exr.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt16, True)
            
            return png_out_model, mesh_center.numpy(), scale_center 
    
    def save_joints(self, format_name: str):
        if format_name.lower() == 'openpose18':
            ...
        elif format_name.lower() == 'smplx':
            ...
    def save_point_cloud(self, format_name: str):
        if format_name.lower() == 'openpose18':
            ...
        elif format_name.lower() == 'smplx':
            ...