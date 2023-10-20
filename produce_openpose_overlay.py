import cv2
import numpy as np
from pathlib import Path

imgs_path = Path('/media/notingcode/Data/Projects/3d_visualize/Images/thuman_multiview_renders').glob(r"[0-9][0-9][0-9][0-9].png")

imgs_list = []

for img_path in imgs_path:   
    if img_path.stem not in imgs_list:
        imgs_list.append(img_path.stem)

for imgs_name in imgs_list:
    img1 = cv2.imread(f'/media/notingcode/Data/Projects/3d_visualize/Images/thuman_multiview_renders/{imgs_name}_front.png')
    img2 = cv2.imread(f'/media/notingcode/Data/Projects/3d_visualize/Images/thuman_multiview_renders/{imgs_name}_back.png')
    img3 = cv2.imread(f'/media/notingcode/Data/Projects/3d_visualize/Images/thuman_multiview_renders/{imgs_name}_left.png')
    img4 = cv2.imread(f'/media/notingcode/Data/Projects/3d_visualize/Images/thuman_multiview_renders/{imgs_name}_right.png')

    result = cv2.hconcat([img1, img2, img3, img4])

    # cv2.imwrite('/media/notingcode/Data/Projects/3d_visualize/Images/openpose_results/concat_poses.png', img=result)

    # src = cv2.imread('/media/notingcode/Data/Projects/3d_visualize/Images/openpose_results/concat_poses.png', 1)
    tmp = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b,g,r = cv2.split(result)

    rgba = [b,g,r,alpha]
    joints = cv2.merge(rgba)

    model_img = cv2.imread(f'/media/notingcode/Data/Projects/3d_visualize/Images/thuman_multiview_renders/{imgs_name}.png')

    b2,g2,r2 = cv2.split(model_img)
    new_alpha = np.full_like(b2,255)
    model_rgba = [b2,g2,r2,new_alpha]

    model_img_alpha = cv2.merge(model_rgba)

    result = cv2.add(model_img_alpha,joints)

    cv2.imwrite(f'./takehome/{imgs_name}_modified.png',result)