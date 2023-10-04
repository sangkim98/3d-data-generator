import open3d as o3d
import mitsuba as mi

def render_mesh(mesh, mesh_center):
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
            'filename': '/home/notingcode/Projects/3d_visualize/polyhaven_hdris/brown_photostudio_03_8k.exr',
            'scale': 1.0,
        },
        'sensor': {
            'type': 'batch',
            'sensor1': {
                'type': 'perspective',
                'fov': 35,
                'to_world': mi.ScalarTransform4f.look_at(origin=[0,0.6+mesh_center[1],2],
                                                         target=mesh_center,
                                                         up=[0,1,0]
                                                         )
            },
            'sensor2': {
                'type': 'perspective',
                'fov': 35,
                'to_world': mi.ScalarTransform4f.look_at(origin=[0,0.6+mesh_center[1],-2],
                                                         target=mesh_center,
                                                         up=[0,1,0]
                                                         )
            },
            'sensor3': {
                'type': 'perspective',
                'fov': 35,
                'to_world': mi.ScalarTransform4f.look_at(origin=[2,0.6+mesh_center[1],0],
                                                         target=mesh_center,
                                                         up=[0,1,0]
                                                         )
            },
            'sensor4': {
                'type': 'perspective',
                'fov': 35,
                'to_world': mi.ScalarTransform4f.look_at(origin=[-2,0.6+mesh_center[1],0],
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

def main():
    mi.set_variant('cuda_ad_rgb')
    
    mesh = o3d.io.read_triangle_mesh('Thuman2_data/Thuman2_SMPL_fittings/THuman2.0_Release/0000/0000.obj', True)
    mesh.textures = [o3d.io.read_image('/home/notingcode/Projects/3d_visualize/Thuman2_data/Thuman2_SMPL_fittings/THuman2.0_Release/0000/material0.jpeg')]
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh_center = tensor_mesh.get_axis_aligned_bounding_box().get_center()
    
    mi_mesh = tensor_mesh.to_mitsuba('thuman')
    img = render_mesh(mi_mesh, mesh_center.numpy())
    png_out = mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt16, True)
    png_out.write('test4.png')
    
main()