import trimesh
import numpy as np

# OBJ 파일을 로드하여 3D 모델 생성
path = "./"
mesh = trimesh.load(path + "frame101.obj")

points, idx = trimesh.sample.sample_surface(mesh, 500)
point_cloud = trimesh.points.PointCloud(points)
point_cloud.export("points.obj")

center = mesh.centroid

print(center)

import pyrender
mesh = pyrender.Mesh.from_trimesh(mesh)
scene = pyrender.Scene()
scene.add(mesh)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = np.sqrt(2)/2

camera_pose = np.array([
   [1.0,  0.0, 0.0, 0 + center[0]],
   [0.0,  1.0, 0.0, 0 + center[1]],
   [0.0,  0.0, 1.0, 2 + center[2]],
   [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(1024, 1024)
color, depth = r.render(scene)

import cv2
cv2.imwrite("a.png", color)




# 특정각도로 카메라를 회전하는 예시

def rotation_matrix_x(angle_rad):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])

def rotation_matrix_y(angle_rad):
    return np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

def rotation_matrix_z(angle_rad):
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])


angle_x_rad = 45 * np.pi / 180
angle_y_rad = 45 * np.pi / 180
angle_z_rad = 30 * np.pi / 180

R_x = rotation_matrix_x(angle_x_rad)
R_y = rotation_matrix_y(angle_y_rad)
R_z = rotation_matrix_z(angle_z_rad)


R = np.dot(R_z, R_y, R_x)

# Rotating the camera's orientation
camera_pose[:3, :3] = np.dot(R, camera_pose[:3, :3])

# Rotating the camera's position
old_position = np.array([0, 0, 2])
new_position = np.dot(R, old_position)
camera_pose[:3, 3] = new_position + center



scene.clear()
scene.add(mesh)
scene.add(camera, pose=camera_pose)
scene.add(light, pose=camera_pose)
color, depth = r.render(scene)
cv2.imwrite("b.png", color)




point_mesh = pyrender.Mesh.from_points(points)
scene.clear()
scene.add(point_mesh)
scene.add(camera, pose=camera_pose)
scene.add(light, pose=camera_pose)
color, depth = r.render(scene)
cv2.imwrite("c.png", color)
