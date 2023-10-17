import cv2
import numpy as np

img1 = cv2.imread('/media/notingcode/Data/Projects/3d_visualize/Images/openpose_results/test4_front.png')
img2 = cv2.imread('/media/notingcode/Data/Projects/3d_visualize/Images/openpose_results/test4_back.png')
img3 = cv2.imread('/media/notingcode/Data/Projects/3d_visualize/Images/openpose_results/test4_left.png')
img4 = cv2.imread('/media/notingcode/Data/Projects/3d_visualize/Images/openpose_results/test4_right.png')

result = cv2.hconcat([img1, img2, img3, img4])

cv2.imwrite('/media/notingcode/Data/Projects/3d_visualize/Images/openpose_results/concat_poses.png', img=result)

src = cv2.imread('/media/notingcode/Data/Projects/3d_visualize/Images/openpose_results/concat_poses.png', 1)
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
b,g,r = cv2.split(src)

rgba = [b,g,r,alpha]
joints = cv2.merge(rgba)

model_img = cv2.imread('/media/notingcode/Data/Projects/3d_visualize/Images/openpose_results/test4.png')

b2,g2,r2 = cv2.split(model_img)
new_alpha = np.full_like(b2,255)
model_rgba = [b2,g2,r2,new_alpha]

model_img_alpha = cv2.merge(model_rgba)

result = cv2.add(model_img_alpha,joints)

cv2.imwrite('./modified.png',result)