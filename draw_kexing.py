import cv2
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import leastsq
import time
import ransac_point as rp
import binary_point as bp
import pcl

CV_LOAD_IMAGE_UNCHANGED  =-1
CV_LOAD_IMAGE_GRAYSCALE  =0
CV_LOAD_IMAGE_COLOR      =1
CV_LOAD_IMAGE_ANYDEPTH   =2
CV_LOAD_IMAGE_ANYCOLOR   =4

# # Depth Scale is 0.001
# intrinsic of camera of (480,640)
# fx = 385.5589294433594
# fy = 385.5589294433594
# ppx = 322.3592529296875
# ppy = 244.20716857910156
fx = 617.686
fy = 618.223
ppx = 317.893
ppy = 250.549

camera_intrinsics = np.array([[fx,0,ppx],[0,fy,ppy],[0,0,1]])
intrinsic_ni = np.linalg.inv(camera_intrinsics)
depth_scale = 0.001
max_len = bp.max_len
act_len = max_len/depth_scale

# transfer matrix to world coordinate system
rate = bp.rate
Rab = bp.Rab

# save_dir = "outdoor_road_depth2"
# img1 = cv2.imread("thir_seg_rem"+"/"+"outdoor_road_color2_43.png",CV_LOAD_IMAGE_COLOR)
# img2 = cv2.imread(save_dir+"/"+"43.png",CV_LOAD_IMAGE_ANYDEPTH)
# img3 = cv2.imread("thir_seg_pic"+"/"+"outdoor_road_color2_43.jpg",CV_LOAD_IMAGE_COLOR)

save_dir = "outstair_depth1"
img1 = cv2.imread("../fifth_rem/outstair1"+"/"+"11.png",CV_LOAD_IMAGE_COLOR)
img2 = cv2.imread(save_dir+"/"+"11.png",CV_LOAD_IMAGE_ANYDEPTH)
img3 = cv2.imread("outstair_color1"+"/"+"11.jpg")

# img2 = cv2.GaussianBlur(img2,(5,5),0)
# img4 = cv2.imread("."+"/"+"label.png",CV_LOAD_IMAGE_COLOR)
# cv2.imshow("origin",img4)
# key = cv2.waitKey(0)

point_matrix = bp.point_transfer(img2)
point = bp.Point_Splice(100,500,point_matrix).astype(np.float32)

# 2 xuan 1,xia mian de  
# cloud = pcl.PointCloud()
# cloud.from_array(point[:,0:3])
# passthrough = cloud.make_passthrough_filter()
# passthrough.set_filter_field_name("z")
# passthrough.set_filter_limits(0.1, max_len)
# cloud = passthrough.filter()
# passthrough = cloud.make_passthrough_filter()
# passthrough.set_filter_field_name("x")
# passthrough.set_filter_limits(-0.25, 0.25)
# cloud = passthrough.filter()
# fil = cloud.make_statistical_outlier_filter()
# fil.set_mean_k(50)
# fil.set_std_dev_mul_thresh(1)
# cloud = fil.filter()
# point1 = np.asarray(cloud)
# tmp = point[:,0:3]
# result = []
# for s in point1:
#     idx = np.argwhere([np.all((tmp-s)==0, axis=1)])[0][1]
#     result.append(idx)
# point = (point[result]).T
# point1 = (np.dot(point1,Rab)).T
# point[0:3] = point1
# point[:,0:3] = np.dot(point[:,0:3],Rab)

# this run more fast than the pcl code, but not include filtering 
im = point[:,0] <= 0.3
jm = point[:,0] >= -0.3
im = im&jm
point = point[im]
point = point.T
point = bp.delete_OutBoundaryPoint(point)

# new a zero imgage
img_z = np.zeros([img1.shape[0],img1.shape[1],3],dtype= np.uint8)

# # de mean draw point edge 
# de = bp.search_edge_point(point)
de  = bp.search_edge_point_simplify(point)

area_de = int(de[1,0]-de[0,0])*int(de[1,1]-de[2,1])
print("Perception area : "+str(area_de) )

img_c = bp.draw_lines(img3,de)
# cv2.imshow("bianjie",img_c)
# # cv2.imshow("origin",img3)
# key = cv2.waitKey(0)

# if want to improve the roi,modify the program array_slice()
# img_z = np.full((img1.shape[0],img1.shape[1],3),120,dtype= np.uint8)
img_z = bp.array_slice(img1,de[0][0],de[1][0],de[2][1],de[0][1],img_z)

kernel = np.ones((5,5),np.uint8)
edges = cv2.Canny(img_z,0,200)
edges = cv2.dilate(edges,kernel,iterations = 1)
edges = cv2.erode(edges,kernel,iterations = 1)
ret,thresh = cv2.threshold(edges,119,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# draw all the picture
thresh = cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
# thresh = cv2.drawContours(thresh, contours, 3, (0,255,0), 3)
# edges = thresh
img1,scene_txt = bp.scene_judge(img_z,contours,area_de)
print(scene_txt)

font=cv2.FONT_HERSHEY_SIMPLEX
txt = "Result:obstacle"
cv2.putText(img1,txt,(10,50), font, 1,(127,255,0),2)

cv2.imshow("bianjie",img_z)
cv2.imshow("canny",img1)
cv2.imshow("edge",edges)
key = cv2.waitKey(0)
