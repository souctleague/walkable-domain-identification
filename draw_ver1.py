import cv2
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import leastsq
import time
import ransac_point_ver2 as rp
import binary_point as bp
import pcl
import copy

def cvtDepth2Cloud(depth, cameraMatrix):
    inv_fx = 1.0 / cameraMatrix[0, 0]
    inv_fy = 1.0 / cameraMatrix[1, 1]
    ox = cameraMatrix[0, 2]
    oy = cameraMatrix[1, 2]

    rows, cols = depth.shape
    cloud = np.zeros((depth.size, 3), dtype=np.float32)
    for y in range(rows):
        for x in range(cols):
            x1 = float(x)
            y1 = float(y)
            dist = depth[y][x]
            cloud[y * cols + x][0] = np.float32((x1 - ox) * dist * inv_fx)
            cloud[y * cols + x][1] = np.float32((y1 - oy) * dist * inv_fy)
            cloud[y * cols + x][2] = np.float32(dist)

    return cloud

def cvtDepthColor2Cloud(depth, color, cameraMatrix):
    inv_fx = 1.0 / cameraMatrix[0, 0]
    inv_fy = 1.0 / cameraMatrix[1, 1]
    ox = cameraMatrix[0, 2]
    oy = cameraMatrix[1, 2]

    rows, cols = depth.shape
    cloud = np.zeros((depth.size, 4), dtype=np.float32)
    for y in range(rows):
        for x in range(cols):

            x1 = float(x)
            y1 = float(y)
            dist = -depth[y][x]
            cloud[y * cols + x][0] = -np.float32((x1 - ox) * dist * inv_fx)
            cloud[y * cols + x][1] = np.float32((y1 - oy) * dist * inv_fy)
            cloud[y * cols + x][2] = np.float32(dist)
            red = color[y][x][2]
            green = color[y][x][1]
            blue = color[y][x][0]
            rgb = np.left_shift(red, 16) + np.left_shift(green,
                                                         8) + np.left_shift(blue, 0)
            cloud[y * cols + x][3] = rgb

    return cloud

CV_LOAD_IMAGE_UNCHANGED  =-1
CV_LOAD_IMAGE_GRAYSCALE  =0
CV_LOAD_IMAGE_COLOR      =1
CV_LOAD_IMAGE_ANYDEPTH   =2
CV_LOAD_IMAGE_ANYCOLOR   =4

camera_intrinsics = bp.camera_intrinsics
intrinsic_ni = np.linalg.inv(camera_intrinsics)
depth_scale = 0.001
max_len = bp.max_len
act_len = max_len/depth_scale
bh = rp.body_high

# Ta = bp.Ta
# Tb = bp.Tb
rate = bp.rate
# cosa = math.cos(Ta)
# sina = math.sin(Ta)
# Ra = np.array([[1,0,0],[0,cosa,-sina],[0,sina,cosa]],dtype= float) 
Rab = bp.Rab
Rba = np.linalg.inv(Rab)

# downstair pwd outdoor_stair_depth2 43.png Ta=30 thred=1.0
# max_len = 2.5
# upstair pwd cy_stair_depth2 15.png Ta=22 thred=1.0
# max_len = 2.5
# slope pwd cy_slope_depth1 200.png Ta=30 thred = 1.0
# max_len = 2.2
# barrier pwd indoor_depth1 70.png Ta=24 thred=1.2
# max_len=3.0
# obstacle pwd indoor_barrier_depth4 114.png Ta=20 thred=1.6
# max_len=3.0

start = time.time()
name,ind,nub = "outdoor","2","62"
depth_dir = name+"_depth"+ind
color_dir = name+"_color"+ind
img1 = cv2.imread(color_dir+"/"+nub+".jpg")
img2 = cv2.imread(depth_dir+"/"+nub+".png",CV_LOAD_IMAGE_ANYDEPTH)
# img2 = cv2.GaussianBlur(img2,(5,5),0)

depth = np.float32(img2)*depth_scale
point = cvtDepth2Cloud(depth,camera_intrinsics)
# point = cvtDepthColor2Cloud(depth,img1,camera_intrinsics)
cloud = pcl.PointCloud()
cloud.from_array(point)
# img2 = cv2.GaussianBlur(img2,(5,5),0)

passthrough = cloud.make_passthrough_filter()
passthrough.set_filter_field_name("z")
passthrough.set_filter_limits(0.1, max_len)
cloud = passthrough.filter()
passthrough = cloud.make_passthrough_filter()
passthrough.set_filter_field_name("x")
passthrough.set_filter_limits(-0.55, 0.55)
cloud = passthrough.filter()

fil = cloud.make_statistical_outlier_filter()
fil.set_mean_k(50)
fil.set_std_dev_mul_thresh(1)
cloud = fil.filter()

sor = cloud.make_voxel_grid_filter()
sor.set_leaf_size(0.02, 0.02, 0.005)
cloud = sor.filter()

point1 = np.asarray(cloud)
point1 = point1.T
point1 = np.dot(Rab,point1)
point1[1] = bh-point1[1]

passthrough = cloud.make_passthrough_filter()
passthrough.set_filter_field_name("x")
passthrough.set_filter_limits(-0.3, 0.3)
cloud = passthrough.filter()

point = np.asarray(cloud)
point = point.T
point = np.dot(Rab,point)
point_bak = point
point_bak = point_bak[:,np.argsort(point_bak[2,:])]

w_min = point[2].min()
h_tmp = -(bh-point[1].max())*rate
# h_plane = h_tmp if h_tmp > 0 else 0
h_plane = h_tmp
point[1] = -(point[1] - point[1].max())
point[2] = point[2] - point[2].min()
point = point*rate
point = point.astype(np.int)
huofu = np.zeros([point[1].max()+1,point[2].max()+1,1], dtype=np.uint8)
huofu = bp.PointToBinary(point,huofu)
huofu = huofu[::-1,:]
h_plane = huofu.shape[0]-h_plane - 5

insert_row = np.zeros([1,point[2].max()+1,1], dtype=np.uint8)
for i in range(0,5):
    huofu = np.insert(huofu, 0, values=insert_row, axis=0)
insert_col = np.zeros([1,1,1], dtype=np.uint8)
for i in range(0,5):
    huofu = np.insert(huofu, 0, values=insert_col, axis=1)
# huofu = np.insert(huofu, huofu.shape[0], values=insert_row, axis=0)

gas = np.append(huofu,huofu,axis = 2)
gas = np.append(gas,huofu,axis = 2)
# blur = cv2.GaussianBlur(gas,(5,5),0)
# xingtaixue bianhua
kernel = np.ones((5,5),np.uint8)
blur = cv2.dilate(gas,kernel,iterations = 1)
blur = cv2.erode(blur,kernel,iterations = 1)
edge = cv2.Canny(blur, 255, 255)

fir_edge_cord,fir_edge = bp.Get_OneEdge(edge,1)
ransac = rp.ransac(fir_edge_cord,w_min,h_plane,rate)
ransac_1 = copy.deepcopy(ransac)

# connect_ransac = rp.connect_ransac(ransac,2)
# print stair parameter
# if connect_ransac.size != 0:
#     print("stair high",connect_ransac[1][1]-connect_ransac[3][3])
#     print("stair kuan",rp.Get_len(connect_ransac[3]))

img3 = rp.draw_walkarea_withlabel(point1,ransac,img1,camera_intrinsics,Rba,w_min)

end = time.time()
print("use time:",end-start)

# blue 1f77b4ff orange ff7f0eff
for i in ransac_1:
    # pos change
    x = np.array([i[0]+w_min*100,i[2]+w_min*100])
    y = np.array([i[1],i[3]])
    # x = np.array([i[0],i[2]])
    # y = np.array([i[1],i[3]])

    plt.plot(x,y,color="#1f77b4ff")
plt.title('stair_para')
plt.xlabel('x/cm')
plt.ylabel('y/cm')
plt.show()


cv2.namedWindow('edge', cv2.WINDOW_AUTOSIZE)

# grad_ruler =  bp.generate_GradColorRuler()
# cv2.imshow('ruler', grad_ruler)

cv2.imshow("kexing",img3)

# cv2.imshow("RGB",img1)
# # cv2.imshow('color', color_grad)
# # cv2.imshow('sobely', sobely)

cv2.imshow('edge', edge)
# cv2.imshow("blur",blur)
cv2.imshow('first_edge', fir_edge)
# cv2.imshow("point",huofu)
# key = cv2.waitKey(0)

# Keep consistent with ankle coordinates
# point_bak[1] = 1.17-point_bak[1]

fig = plt.figure()
ax1 = plt.axes(projection='3d')
# draw
ax1.scatter3D(point1[0],point1[2],point1[1], cmap='Blues',s= 5)  
# ax1.plot3D(x,y,z,'gray')        
plt.show()