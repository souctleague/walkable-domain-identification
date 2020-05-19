import cv2
import os
import numpy as np
import math
import time
import linecache
import re

CV_LOAD_IMAGE_UNCHANGED  =-1
CV_LOAD_IMAGE_GRAYSCALE  =0
CV_LOAD_IMAGE_COLOR      =1
CV_LOAD_IMAGE_ANYDEPTH   =2
CV_LOAD_IMAGE_ANYCOLOR   =4

# D435
# fx = 384.402
# fy = 384.402
# ppx = 318.067
# ppy = 240.294

# D435i
fx = 611.821
fy = 610.281
ppx = 314.649
ppy = 238.833

camera_intrinsics = np.array([[fx,0,ppx],[0,fy,ppy],[0,0,1]])
intrinsic_ni = np.linalg.inv(camera_intrinsics)
depth_scale = 0.001
max_len = 3.5
act_len = max_len/depth_scale
rate = 200

Ta = math.radians(-3.84)
Tb = math.radians(0.0)
Tc = math.radians(0.0)

cosa = math.cos(Ta)
sina = math.sin(Ta)
Rx = np.array([[1,0,0],[0,cosa,-sina],[0,sina,cosa]],dtype= float) 
cosb = math.cos(Tb)
sinb = math.sin(Tb)
Rz = np.array([[cosb,-sinb,0],[sinb,cosb,0],[0,0,1]],dtype= float) 
cosc = math.cos(Tc)
sinc = math.sin(Tc)
Ry = np.array([[cosc,0,-sinc],[0,1,0],[sinc,0,cosc]],dtype= float)
Rab = np.dot(Rz,Rx)
# street
# Ta = math.radians(24.0344026)
# cosa = math.cos(Ta)
# sina = math.sin(Ta)
# Rab = np.array([[1,0,0],[0,cosa,-sina],[0,sina,cosa]],dtype= float) 

def Camera_coordinate_To_World(angle = np.array([0.0,0.0,0.0])):
    if angle.size != 3:
        print("Input angle have some trouble")
        angle = np.array([0.0,0.0,0.0])
    else:
        angle = angle.reshape(3,)
    
    Ta =  math.radians(angle[2])
    Tb =  math.radians(angle[0])
    Tc =  math.radians(angle[1])

    cosa = math.cos(Ta)
    sina = math.sin(Ta)
    Rx = np.array([[1,0,0],[0,cosa,-sina],[0,sina,cosa]],dtype= float) 
    cosb = math.cos(Tb)
    sinb = math.sin(Tb)
    Rz = np.array([[cosb,-sinb,0],[sinb,cosb,0],[0,0,1]],dtype= float) 
    cosc = math.cos(Tc)
    sinc = math.sin(Tc)
    Ry = np.array([[cosc,0,-sinc],[0,1,0],[sinc,0,cosc]],dtype= float)
    # don't use Ry
    Rab = np.dot(Rz,Rx)
    Rba = np.linalg.inv(Rab)

    Rab = Rab.astype(np.float32)
    Rba = Rba.astype(np.float32)

    return Rab,Rba

def delete_point(draw_point):
    im = draw_point[:,2] < max_len
    draw_point = draw_point[im]
    return draw_point
    
def delete_OutBoundaryPoint(point):
    im = point[1,:] != 0
    point = point[:,im]
    return point

def from_pixel_to_point(flip,num):
    res_camera = np.ones((3,480),dtype = np.float)
    tmp = list(range(flip.shape[0]))
    res_camera[1] = tmp
    res_camera[0] = num
    res_camera = res_camera*(flip/1000.0)
    res_camera = np.dot(intrinsic_ni,res_camera)
    return res_camera.T

def search_deep_point(p,p_max):
    l = p_max[0].min()
    r = p_max[0].max()

    ind = np.argwhere(p_max[0] < l+0.1)
    res = p_max[4,ind]
    l_col = res.min()
    l_ind = np.argwhere(p_max[4] == l_col)
    l_row = p_max[3,ind].min()    
    
    ind = np.argwhere(p_max[0] > r-0.1)
    res = p_max[4,ind]
    r_col = res.max()
    r_ind = np.argwhere(p_max[4] == r_col)
    r_row = p_max[3,ind].min() 

    ret = np.array([[l_row,l_col],[r_row,r_col]])
    return ret

def search_short_point(p,p_min):
    l = p_min[0].min()
    r = p_min[0].max()

    ind = np.argwhere(p_min[0] < l+0.1)
    res = p_min[4,ind]
    l_col = res.min()
    l_ind = np.argwhere(p_min[4] == l_col)
    l_row = p_min[3,ind].max()    
    
    ind = np.argwhere(p_min[0] > r-0.1)
    res = p_min[4,ind]
    r_col = res.max()
    r_ind = np.argwhere(p_min[4] == r_col)
    r_row = p_min[3,ind].max() 

    ret = np.array([[l_row,l_col],[r_row,r_col]])
    return ret

def search_edge_point(p):
    mi = p[2].min()
    ma = p[2].max()
    im = p[2] < mi+0.2
    p_mi = p[:,im]
    ret_mi = search_short_point(p,p_mi)
    
    im = p[2] > ma-0.2
    p_ma = p[:,im]
    ret_ma = search_deep_point(p,p_ma)

    if ret_ma[0][0] > ret_ma[1][0]:
        ret_ma[0][0] = ret_ma[1][0]
    else:
        ret_ma[1][0] = ret_ma[0][0]

    # ret_ma[0][1],ret_ma[1][1] = ret_mi[0][1],ret_mi[1][1]

    ret = np.vstack((ret_mi,ret_ma))
    ret = ret.astype(np.int16)
    ret = ret[:,::-1]
    return ret

# walkarea get by point tranfer to pixel
def search_edge_reverse(p,r):
    # pl = p[0].min()
    # pr = p[0].max()

    im = p[2,:] <= p[2].min() + 0.15
    p_tmp = p[:,im]
    ph = p_tmp[1].max()

    p1 = np.array([-0.3,ph,p[2].min()])
    p2 = np.array([0.3,ph,p[2].min()])
    p1,p2 = np.dot(r,p1),np.dot(r,p2)
    p1,p2 = np.dot(camera_intrinsics,p1),np.dot(camera_intrinsics,p2)
    p1,p2 = (p1/p1[2]).astype(np.int16),(p2/p2[2]).astype(np.int16)

    im = p[2,:] >= p[2].max() -0.15
    p_tmp = p[:,im]
    ph = p_tmp[1].min()
    
    p3 = np.array([0,ph,p[2].max()])
    p3 = np.dot(r,p3)
    p3 = np.dot(camera_intrinsics,p3)
    p3 = (p3/p3[2]).astype(np.int16)

    pl = p1[0] if p1[0] >= 0 else 0 
    pr = p2[0] if p2[0] <= 640 else 640 
    pw = p1[1] if p1[1] > p2[1] else p2[1]
    pw = pw if pw <= 480 else 480
    ph = p3[1] if p3[1] >= 0 else 0
    if pw < ph :
        tmp = pw
        pw = ph
        ph = tmp
    
    ret = np.array([[pl,pw],[pr,pw],[pl,ph],[pr,ph]],dtype= np.int16)
    return ret
    
# walkarea get by origin point which have pixel postion
def search_edge_point_simplify(p):
    pl = p[4].min()
    pr = p[4].max()
    ph = p[3].min()
    pw = p[3].max()
    ret = np.array([[pw,pl],[pw,pr],[ph,pl],[ph,pr]])
    ret = ret.astype(np.int16)
    ret = ret[:,::-1]
    return ret

def array_slice(p,pl,pr,ph,pw,ret):
    for i in range(ph,pw):
        ret[i,pl:pr] = p[i,pl:pr]
    return ret

def draw_lines(img,d):
    ret = cv2.line(img,tuple(d[0]),tuple(d[1]),(255,0,0),5)
    ret = cv2.line(img,tuple(d[0]),tuple(d[2]),(255,0,0),5)
    # ret = cv2.line(img,tuple(d[0]),tuple(d[3]),(255,0,0),5)
    # ret = cv2.line(img,tuple(d[1]),tuple(d[2]),(255,0,0),5)
    ret = cv2.line(img,tuple(d[1]),tuple(d[3]),(255,0,0),5)
    ret = cv2.line(img,tuple(d[2]),tuple(d[3]),(255,0,0),5)
    return ret

#Camera Coordinate System transfer
def point_transfer(img):
    h,w = img.shape[0],img.shape[1]
    res = np.ones((h,w,3),dtype = np.float32)
    for i in range(0,w):
        res[:,i] = from_pixel_to_point(img[:,i],i)
    return res

def Point_Splice(beg,end,ori):
    len = ori.shape[0]
    res = np.array([],dtype= np.float32)

    tmp = ori[:,beg]
    col = np.empty(len)
    col.fill(beg)
    row = np.arange(0,len,1)
    tmp = np.column_stack((tmp,row,col))
    tmp = delete_point(tmp)
    res = tmp
    for i in range(beg+1,end+1):
        tmp = ori[:,i]
        col.fill(i)
        tmp = np.column_stack((tmp,row,col))
        tmp = delete_point(tmp)
        res = np.vstack((res,tmp))
    res = res.astype(np.float32)
    return res


def PointToBinary(point,map):
    nums = point.shape[1]
    for i in range(0,nums):
        map[point[1,i],point[2,i]] = 255
    # new_row = np.zeros((1,map.shape[1],1),dtype= int)
    # map = np.insert(map,0,values= new_row,axis = 0)
    return map

# obtain the gradient value and direction
def GetGrad(x,y):
    Gx = x**2
    Gy = y**2
    G = (Gx + Gy)**0.5
    h,w = Gx.shape[0],Gx.shape[1]
    x = x.reshape(h*w,1)
    y = y.reshape(h*w,1)
    theta = np.zeros((h*w,1),dtype= float)
    color = np.zeros((h*w,3),dtype= np.uint8)
    for i in range(0,x.size):
        if x[i] == 0 and y[i] < 0:
            theta[i] = -90
            color[i] = [125,125,125]
        elif x[i] == 0 and y[i] > 0:
            theta[i] = 90
            color[i] = [0,255,255]
        elif x[i] > 0 and y[i] == 0:
            theta[i] = 0
            color[i] = [255,255,255]
        elif x[i] < 0 and y[i] == 0:
            theta[i] = 180
            [255,255,255]
        else:
            theta[i] = math.degrees(math.atan(y[i]/x[i]))
            if theta[i] == np.nan:
                color[i] = [0,0,0]
            elif theta[i] > 0 and theta[i] < 90:
                color[i] = [0,0,255]
            elif theta[i] >90 and theta[i] < 180:
                color[i] = [0,0,255]
            elif theta[i] > -180 and theta[i] < -90:
                color[i] = [255,255,0]
            elif theta[i] > -90 and theta[i] < 0:
                color[i] = [255,255,0]
    theta = theta.reshape(h,w)
    color = color.reshape(h,w,3)
    # Gx = Gx.reshape(h,w)
    return theta,G,color

def add_row_edge(row,point,color):
    ind = sorted(row)
    for key in ind:
        if key > point[0][0]:
            break
        flag,i = 0,0
        while i < point.shape[0]:
            if point[i][0]== key and point[i][1] == row[key]:
                break
            elif flag != 1 and abs(point[i][0] -key) < 12 and abs(point[i][1]-row[key])<12:
                flag = 1
                if point[i][1] > row[key]:
                    tmp_i = i
                else:
                    tmp_i = i+1
            i = i + 1
        if i >= point.shape[0] and flag == 1:
            point = np.insert(point,tmp_i,[key,row[key]],axis=0)
            color[key][row[key]] = [255,255,255]
        # color[key][row[key]] = [255,255,255]

    return point,color

def step_search(col,row,k_c,edge):
    res = np.array([],dtype= int)
    flag = 0
    for i in k_c:
        if flag != 0:
            flag = flag - 1
            continue
        if col.has_key(i+1):
            res = np.append(res,[col[i],i])
            if col[i +1] - col[i] > 0:
                beg = col[i]
                end = col[i+1]
            else:
                beg = col[i+1]
                end = col[i]
            # col[i] grow to col[i+1]
            last_j = i+1
            for j in range(beg,end):
                if last_j >= i+1 and last_j < edge.shape[1]-1:
                    if edge[j][last_j] != 0:
                        res = np.append(res,[j,last_j])
                        last_j = last_j
                    elif edge[j][last_j-1] != 0:
                        res = np.append(res,[j,last_j-1])
                        last_j = last_j-1
                    elif edge[j][last_j+1] != 0:
                        res = np.append(res,[j,last_j+1])
                        last_j = last_j+1
                    # elif row.has_key(j) and col.has_key(row[j])!= j\
                    #      and abs(row[j]-last_j<=3):
                    #     res = np.append(res,[j,row[j]])
                    #     last_j = row[j]
                    #     row[j] = 99999999
                elif last_j < i+1:
                    if edge[j][last_j+1] != 0:
                        res = np.append(res,[j,last_j+1])
                        last_j = last_j+1
                    elif edge[j][last_j-1] != 0:
                        res = np.append(res,[j,last_j-1])
                        last_j = last_j-1
                    # elif row.has_key(j) and col.has_key(row[j])!= j\
                    #      and abs(row[j]-last_j<=3):
                    #     res = np.append(res,[j,row[j]])
                    #     last_j = row[j]
                    #     row[j] = 99999999
                elif last_j < edge.shape[1]-2 and edge[j][i+1] != 0:
                    res = np.append(res,[j,i+1])
                    last_j = i+1
                elif last_j < edge.shape[1]-3 and edge[j][i+2] != 0:
                    res = np.append(res,[j,i+2])
                    last_j = i+2
                elif edge[j][i] != 0:
                    res = np.append(res,[j,i])
                    last_j = i 
        elif col.has_key(i) == False and col.has_key(i+2):
            if abs(col[i+2] - col[i]) <= 2:
                res = np.append(res,[col[i+2],i+2])
                flag  = 1
        elif col.has_key(i) == False and col.has_key(i+3):
            if abs(col[i+3] - col[i]) <= 2:
                res = np.append(res,[col[i+3],i+3])
                flag = 2
        elif col.has_key(i) == False and col.has_key(i+4):
            if abs(col[i+4] - col[i]) <= 2:
                res = np.append(res,[col[i+4],i+4])
                flag = 3
        else:
            res_i = i
            break
    res = res.reshape(res.size/2,2)
    return res,res_i
            

def Recur_Get_Edge(edge,k_c,col,row,res):
    res = list()
    k = k_c[0]
    while k <= k_c[-1]:
        if col.has_key(k) == False:
            k = k+1
            continue
        if col.has_key(k+1) and col.has_key(k+2)\
             and abs(col[k+1]-col[k])<=1 and abs(col[k+2]-col[k+1])<=1:
            tmp_edge = np.array([],dtype= int)
            ind = k_c.index(k)
            tmp_k = k_c[ind:]
            if k < k_c[-10] :
                tmp_edge,k =  step_search(col,row,tmp_k,edge)
                res.append(tmp_edge)
        if k == k_c[-1] or k == k_c[-2]:
            break
        # ind = k.index(k)
        k = k+1
    return res

# obstacle scene type_walk = 1 , other = 0
def Get_OneEdge(edge,type_walk=0):
    res = []
    row = {}
    for i in range(0,edge.shape[0]):
        for j in range(0,edge.shape[1]):
            if edge[i][j] == 255:
                row[i] = j
                break
        # if j == edge.shape[1] - 1 and edge[i][j] != 255:
        #     row[i] = np.nan
    col = {}
    for i in range(0,edge.shape[1]):
        for j in range(0,edge.shape[0]):
            if edge[j][i] == 255:
                col[i] = j
                break
        # if j == edge.shape[0] - 1 and edge[j][i] != 255:
        #     col[i] = np.nan
   
    k_c = sorted(col)
    tmp_res = Recur_Get_Edge(edge,k_c,col,row,res)
    if tmp_res != []:
        res = tmp_res    
    
    color = np.zeros((edge.shape[0],edge.shape[1],3),dtype= np.uint8)
    for i in range(0,len(res)):
        tmp_res = res[i]
        if i == 0:
            merg = tmp_res
        else:
            merg =  np.concatenate((merg,tmp_res),axis=0)
        for j in range(0,tmp_res.shape[0]):
            x,y = tmp_res[j]
            # color[x][y] = [255-50*i,255-50*i,255-50*i]
            color[x][y] = [255,255,255]
    

    if type_walk == 1:
        merg,color = add_row_edge(row,merg,color)
    
    return merg,color


def generate_GradColorRuler():
    CLASSES = open("txt/classes.txt").read().strip().split("\n")
    COLORS = open("txt/colors.txt").read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")
    # initialize the legend visualization
    legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")
    # loop over the class names + colors
    for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
        # draw the class name + color on the legend
        color = [int(c) for c in color]
        cv2.putText(legend, className, (5, (i * 25) + 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(legend, (120, (i * 25)), (300, (i * 25) + 25),
            tuple(color), -1)
    return legend

def print_ColorAndTxt(edge,theta):
    color_grad = np.zeros((edge.shape[0],edge.shape[1],3),dtype= np.uint8)
    f = open('txt/grad.txt','w+')
    for i in range(0,edge.shape[0]):
        for j in range(0,edge.shape[1]):
            if edge[i][j] == 255:
                tmp_s = "["+str(i)+"]"+"["+str(j)+"] ="+str(theta[i][j]) +"  " 
                f.write(tmp_s)
                if theta[i][j] > 0 and theta[i][j] < 90:
                    #red
                    color_grad[i][j] = [0,0,255]
                elif theta[i][j] >90 and theta[i][j] <= 180:
                    #blue
                    color_grad[i][j] = [255,0,0]
                elif theta[i][j] >= -180 and theta[i][j] < -90:
                    # little blue
                    color_grad[i][j] = [255,255,0]
                elif theta[i][j] >= -90 and theta[i][j] < 0:
                    #green
                    color_grad[i][j] = [0,255,0]
                elif theta[i][j] == 90 or theta[i][j] == -90:
                    #yellow
                    color_grad[i][j] = [0,255,255]
                elif theta[i][j] == 180 or theta[i][j] == 0:
                    #purple
                    color_grad[i][j] = [0,215,255]
                continue
            if math.isnan(theta[i][j]) != True:
                tmp_s = "["+str(i)+"]"+"["+str(j)+"] ="+str(theta[i][j])+"  " 
                f.write(tmp_s)
    f.close()
    return color_grad

def ransac_scene(txt,ransac):
    if txt == "outdoor" or txt == "indoor":
        return 0
    elif txt == "obstacle":
        return 3
    elif txt == "stair":
        if ransac[0][1] >ransac[-1][3]:
            return 2
        elif ransac[0][1] <= ransac[-1][3]:
            return 1

def draw_walk(img,point,R,draw_img = None):
    im = point[0,:] <= 0.3
    jm = point[0,:] >= -0.3
    im = im&jm
    point = point[:,im]

    img_z = np.zeros([img.shape[0],img.shape[1],3],dtype= np.uint8)
    de  = search_edge_reverse(point,R)
    area_de = int(de[1,0]-de[0,0])*int(de[1,1]-de[2,1])
    # print("Perception area : "+str(area_de) )
    if draw_img != None:
        draw_img = draw_lines(img3,de)
    
    img_z = array_slice(img,de[0][0],de[1][0],de[2][1],de[0][1],img_z)

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
    img,scene_txt = scene_judge(img_z,contours,area_de)

    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,scene_txt,(10,50), font, 1,(127,255,0),2)

    return img, scene_txt, draw_img

def scene_judge(img,cntours,area,deal_type = 1):
    
    def Get_scene_color(img,x,y,w,h):
        tmp_img = img[y:y+h,x:x+w]
        tmp_img = tmp_img.reshape(tmp_img.shape[0]*tmp_img.shape[1],3)
        G_count = np.bincount(tmp_img[:,1])
        val = np.argmax(G_count)
        val_area = G_count[val]
        return val,val_area
    
    if deal_type == 0:
        i,j,k = 2,1,0
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(i*50,j*50,k*50),2)
            print("count " + str(j) +" area: "+str(w*h) )
            i,j,k = i+1,j+1,k+1
            # if i < 5:
            #     i = i + 1 
            # elif j < 5:
            #     j = j +1
            # elif k < 5:
            #     k = k +1
            # else:
            #     i,j,k = i-5,j-5,k-5
        res_scene_txt = "no scene type"
    elif deal_type == 1:
        # area divide 5 ~ 10 is all right
        # due to influence of stair scene, choose 5 here
        area_filter,area_res = area*1.0/5,np.zeros(4,dtype= np.int32)
        
        i,j,k = 2,1,0
        for cnt in cntours:
            x,y,w,h = cv2.boundingRect(cnt)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(i*50,j*50,k*50),2)
            i,j,k = i+1,j+1,k+1
            # area_cnt = cv2.contourArea(cnt)
            area_cnt = w*h
            print("count " + str(k) +" area: "+str(area_cnt) )
            if area_cnt > area_filter and area_cnt < area-1000:
                # use result img zhong R or G messege judge scene 
                max_val,max_val_area = Get_scene_color(img,x,y,w,h)
            elif len(cntours) == 2 and abs(area_cnt - area) < 2000:
                max_val,max_val_area = Get_scene_color(img,x,y,w,h)
            elif len(cntours) == 1:
                max_val,max_val_area = Get_scene_color(img,x,y,w,h)
            
            if "max_val" in locals().keys():
                # obstacle
                if max_val == 120 or  max_val == 250:
                    area_res[0] = area_res[0] + max_val_area
                    if area_res[0] >= area*1.0/3:
                        res_scene = [0,]
                #stair 
                elif max_val == 0:
                    area_res[1] = area_res[1] + max_val_area
                    if area_res[1] >= area*1.0/2 and ("res_scene" in locals().keys()) == False or res_scene[0] == 2 \
                        or res_scene[0] == 3:
                        res_scene = [1,]
                # indoor
                elif max_val == 50:
                    area_res[2] = area_res[2] + max_val_area
                    if area_res[2] >= area*2.0/3 and ("res_scene" in locals().keys()) == False or res_scene[0] == 3:
                        res_scene = [2,]
                # outdoor
                elif max_val == 255:
                    area_res[3] = area_res[3] + max_val_area
                    if area_res[3] >= area*2.0/3 and ("res_scene" in locals().keys()) == False:
                        res_scene = [3,]

        # simplify scene judge
        if ("res_scene" in locals().keys()) == False:
            res_scene = (np.argmax(area_res),)
        if res_scene[0] == 3:
            res_scene_txt = "outdoor"
        elif res_scene[0] == 2:
            res_scene_txt = "indoor"
        elif res_scene[0] == 0:
            res_scene_txt = "obstacle"
        elif res_scene[0] == 1:
            res_scene_txt = "stair" 

    return img,res_scene_txt

def make_file(d_name,save_dir):
    work_path = save_dir
    list_num = os.listdir(work_path)
    
    if not list_num:
        os.mkdir(work_path+"/"+d_name+"1")
        return d_name+str(1)
    elif d_name == "":
        list_str = os.listdir(work_path)
        list_num = 0
        for i in list_str: 
            if i.isdigit:
                list_num += 1
        max_dir = list_num + 1
        os.mkdir(work_path+"/"+str(max_dir))
        print("create directory success")
        return str(max_dir)
    else:
        list_str = os.listdir(work_path)
        #print(list_num)
        # list_num = [int(i) for i in list_str]
        # max_dir = max(list_num)
        # max_dir +=1
        max_dir = 1
        for x in list_str :
                # if os.path.isfile(os.path.join(list_str, x)) and d_name in x:
                if d_name in x:
                    cfg_fliter =  filter(str.isdigit, x) 
                    cfg_list = list(cfg_fliter)       
                    cfg_str = "".join(cfg_list)
                    bianhao = int(cfg_str) 
                    if bianhao >= max_dir:
                        max_dir = bianhao +1

        os.mkdir(work_path+"/"+d_name+str(max_dir))
        print("create directory success")
        return d_name+str(max_dir)

def find_imgs(args):
    dir_name = args.imgs
    if dir_name[-3:] == "jpg" or dir_name[-3:] == "png":
        dir_match = re.match(r"(\w+)_[a-zA-Z]+(\d+)/(\d+).[a-zA-Z]",dir_name)
        if dir_match != None:
            name,ind,nub = dir_match.group(1),\
                 dir_match.group(2), dir_match.group(3)
            depth_res = [name+"_depth"+ind+"/"+nub+".png"]
            color_res = [name+"_color"+ind+"/"+nub+".jpg"]
            if args.pos == 1:
                tmp_txt = linecache.getline(name+"_color"+ind+"_angle_data.txt",int(nub))
                tmp_txt = tmp_txt[:-1]
                tmp_pos = re.match("[\d\s]+x: ([\-\d\.]+) y: ([\-\d\.]+) z: ([\-\d\.]+)",tmp_txt)
                pos = np.array([float(tmp_pos.group(1)),float(tmp_pos.group(2)),float(tmp_pos.group(3))])
            else:
                pos = np.array([0,0,0],dtype= np.float)
            # if args.rem[-3:] != "jpg" and args.rem[-3:] != "png":
            #     rem_img_res = args.rem+"/"+nub+".png"
            # else:
            # 
            if args.rem[-3:] != "png" and os.path.isdir(args.rem):
                rem_res = [args.rem+"/"+nub+".png",]
            elif args.rem[-3:] == "png":
                rem_res = [args.rem,]
            else:
                return

            return depth_res,color_res,rem_res,pos
        else:
            return
    else:
        dir_match = re.match(r"([A-Za-z]+)(\d+)",dir_name)
        if dir_match == None:
            return
        elif os.path.isdir(dir_match.group(1)+"_color"+dir_match.group(2)):
            name,ind = dir_match.group(1),dir_match.group(2)
            if args.rem[-3:] == "jpg" or args.rem[-3:] == "png" or os.path.isdir(args.rem) == False:
                print("Please input remove picture's directory")
                return
            else :
                rem_dir = args.rem
                rem_num = []
                rem_img = os.listdir(rem_dir)
                for img in rem_img:
                    rem_files = re.match("([\d]+).png",img)
                    if rem_files != None:
                        rem_num.append(int(rem_files.group(1)))
                rem_num.sort()
                
                depth_dir = name+"_depth"+ind+"/"
                color_dir = name+"_color"+ind+"/"
                txt = name+"_color"+ind+"_angle_data.txt"
                depth_res,color_res,rem_res,pos = [],[],[],np.array([])

                if args.pos == 1:
                    for i in rem_num:
                        depth_res.append(depth_dir+str(i)+".png")
                        color_res.append(color_dir+str(i)+".jpg")
                        rem_res.append(rem_dir+"/"+str(i)+".png")

                        tmp_txt = linecache.getline(txt,i)
                        tmp_txt = tmp_txt[:-1]
                        tmp_pos = re.match("[\d\s]+x: ([\-\d\.]+) y: ([\-\d\.]+) z: ([\-\d\.]+)",tmp_txt)
                        pos = np.append(pos,np.array([float(tmp_pos.group(1)),float(tmp_pos.group(2)),float(tmp_pos.group(3))]))
                    pos = pos.reshape(pos.shape[0]/3,3)
                else:
                    for i in rem_num:
                        depth_res.append(depth_dir+str(i)+".png")
                        color_res.append(color_dir+str(i)+".jpg")
                        rem_res.append(rem_dir+"/"+str(i)+".png")
                        pos = np.append(pos,np.array([0,0,0]))
                    pos = pos.reshape(pos.shape[0]/3,3)
                
                return depth_res,color_res,rem_res,pos
        else :
            return
