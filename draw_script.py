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
import re
import argparse

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
rate = bp.rate
bh = rp.body_high

def main(args):
    start = time.time()

    imgs_depth,imgs_color,imgs_rem,imgs_pos = bp.find_imgs(args)
    if imgs_depth == None:
        print("Please check the directory of RDB-D")
        return
    elif imgs_rem == None:
        print("Please check the directory of RDB-D")
        return
    # .;assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    rem_dir = bp.make_file("rem","result_walkable")
    rem_dir = "result_walkable/"+rem_dir+"/"
    param_dir = bp.make_file("parameter","result_walkable")
    param_dir = "result_walkable/"+param_dir+"/"

    img_num = len(imgs_rem)
    for i in range(0,img_num):
        img1 = cv2.imread(imgs_color[i])
        img2 = cv2.imread(imgs_depth[i],CV_LOAD_IMAGE_ANYDEPTH)
        img3 = cv2.imread(imgs_rem[i])
        imgs_pos[i][0] = 0
        Rab,Rba = bp.Camera_coordinate_To_World(imgs_pos[i])

        # img2 = cv2.GaussianBlur(img2,(5,5),0)
        if (img1 == None).all() or (img2 == None).all() or (img3 == None).all():
            if (img1 == None).all() or (img2 == None).all():
                print(img1 + "is not found")
            else:
                print(img + "is not found")
            continue
        

        depth = np.float32(img2)*depth_scale
        point = cvtDepth2Cloud(depth,camera_intrinsics)
        # point = np.dot(Rab,point.T)
        # point = point.T

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
        point1[1] = bh -point1[1]

        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("x")
        passthrough.set_filter_limits(-0.3, 0.3)
        cloud = passthrough.filter()

        point = np.asarray(cloud)
        point = point.T
        point = np.dot(Rab,point)
        point_bak = copy.deepcopy(point)
        # point_bak = point_bak[:,np.argsort(point_bak[2,:])]
        
        w_min = point[2].min()
        h_tmp = -(bh - point[1].max())*rate
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
        for j in range(0,5):
            huofu = np.insert(huofu, 0, values=insert_row, axis=0)
        insert_col = np.zeros([1,1,1], dtype=np.uint8)
        for j in range(0,5):
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

        img3,scene_txt,img_walk = bp.draw_walk(img3,point_bak,Rba)
        scene_type = bp.ransac_scene(scene_txt,ransac)
        
        # connect_ransac = rp.connect_ransac(ransac,scene_type)
        # need add a scene jugde
        
        # for i in ransac_1:
        #     # pos change
        #     x = np.array([i[0]+w_min*100,i[2]+w_min*100])
        #     y = np.array([i[1],i[3]])
        #     # x = np.array([i[0],i[2]])
        #     # y = np.array([i[1],i[3]])

        #     plt.plot(x,y,color="#1f77b4ff")
        # plt.title('stair_para')
        # plt.xlabel('x/cm')
        # plt.ylabel('y/cm')
        # plt.show()

        img_label = rp.draw_walkarea_withlabel(point1,ransac_1,img1,camera_intrinsics,Rba,w_min)
        cv2.imwrite(rem_dir+str(i)+".jpg",img3)
        cv2.imwrite(param_dir+str(i)+".jpg",img_label)

        
        
        # fig = plt.figure()
        # ax1 = plt.axes(projection='3d')
        # # draw
        # ax1.scatter3D(point1[0],point1[2],point1[1], cmap='Blues',s= 5)  
        # # ax1.plot3D(x,y,z,'gray')        
        # plt.show()

    
    end = time.time()
    print("use time:",end-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", 
        "--rem",
        default = "../fifth_rem/outdoor2", 
        help="add the directory of pictures which removed color or single picture",
        type=str
    )
    parser.add_argument(
        "-i", 
        "--imgs",
        default = "outdoor2", 
        help="the key word of directory.for example \"outdoor1,2...3\" represent outdoor_color_1,2...3",
        type=str
    )
    parser.add_argument(
        "-p", 
        "--pos",
        default = "1", 
        help="1 is Enable Euler Point,0 is don't use Euler angles ",
        type=int
    )
    args = parser.parse_args()
    main(args)