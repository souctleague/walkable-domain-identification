import cv2
import sys
import os
import numpy as np
import glob

CV_LOAD_IMAGE_UNCHANGED  =-1
CV_LOAD_IMAGE_GRAYSCALE  =0
CV_LOAD_IMAGE_COLOR      =1
CV_LOAD_IMAGE_ANYDEPTH   =2
CV_LOAD_IMAGE_ANYCOLOR   =4

# (80,50,50) floor;flooring
# (255,31,0) path (235,255,7) sidewalk;pavement
#  (140,140,140) road;route
# (0,255,163) escalator;moving (31,0,255) staircase;stairway
# (255,224,0) stairs;steps
# (4,250,7) grass (61,230,250) water (120,120,70) ground

def crop_image(img):
    w,h = img.shape[0],img.shape[1]
    for i in range(0,w):
        for j in range(0,h):
            # indoor 
            if (img[i,j] == [50,50,80]).all() or (img[i,j] == [7,255,235]).all()\
                 or (img[i,j] == [255,0,31]).all() or (img[i,j] == [7,250,4]).all():
                continue
            # outdoor pavement
            elif (img[i,j] == [140,140,140]).all() or (img[i,j] == [0,31,255]).all():
                img[i,j] = [7, 255, 235]
            # stair
            elif (img[i,j] == [0,224,255]).all():
                img[i,j] = [255,0,31]
            # non-walkable
            elif (img[i,j] == [163,255,0]).all() or (img[i,j] == [70,120,120]).all():
               img[i,j] = [7,250,4]
            # others
            else :
                img[i,j] = [120,120,120]
         
    return img


read_dir = "error_pic/seg"
write_dir = "error_pic/rev/"

# img = cv2.imread("outdoor_road_color2_43.png")
# output_img = crop_image(img)
# cv2.imwrite(write_dir + "outdoor_road_color2_43.png",output_img)

paths = glob.glob(os.path.join(read_dir, '*.png'))
paths.sort()
j = 1
for path in paths:
    img = cv2.imread(path)
    if img.all() == None:
        print("maybe wrong directory")
        break
    output_img = crop_image(img)
    name = os.path.basename(path)
    cv2.imwrite(write_dir + name,output_img)
    print(name+ " picture is OK")

print("Full of ALL Are OK!")

# j = 1
# while j < 2000:
#     input_img = cv2.imread(save_dir+'%d.png'%j,cv2.IMREAD_COLOR)
#     if input_img.all() == None:
#         break
#     output_img = crop_image(input_img)
#     cv2.imwrite('5_rev/%d.png'%j,output_img)
#     print("number "+ str(j)+ "picture is OK")
#     j += 1
