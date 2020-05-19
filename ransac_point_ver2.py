import numpy as np
import math
import cv2
import copy

body_high = 1.17

def line_intersection(line1, line2):
    k1,k2 = Get_k(line1),Get_k(line2)
    if k1 == k2:
        return line1[2],line1[3]

    if line1.shape[0] == 4:
        line1 = line1.reshape(2,2)
    if line2.shape[0] == 4:
        line2 = line2.reshape(2,2)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here
 

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    a=lineY2-lineY1
    b=lineX1-lineX2
    c=lineX2*lineY1-lineX1*lineY2
    dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
    return dis

def new_kb(point):
    # point = point.reshape((point.size/2,2))
    x = point[:,0]
    y = point[:,1]
    z1 = np.polyfit(x, y, 1)
    k = z1[0]
    b = z1[1]
    return k,b

# improve of new_kb,use ransac idea
def ransac_kb(point,best_k,best_b):
    k,b = best_k,best_b
    bad_point = 0
    thred = 2.5
    best_sum = 0
    for j in range(0,point.size/2):
        dis = abs(getDis(point[j][0],point[j][1],0,b,1,k+b))
        best_sum = best_sum + dis
        if(dis > thred):
            bad_point = bad_point +1
    last_bad_point = bad_point

    for i in range(0,10):
        index = np.random.choice(point.size/2, size=4, replace=False, p=None)
        p = point[index]
        k,b = new_kb(p)
        tmp_sum ,bad_point= 0,0

        for j in range(0,point.size/2):
            dis = abs(getDis(point[j][0],point[j][1],0,b,1,k+b))
            tmp_sum = tmp_sum + dis
            if(dis > thred):
                bad_point = bad_point +1
        if bad_point < last_bad_point:
            best_k,best_b = k,b
            last_bad_point,best_sum = bad_point,tmp_sum
        elif tmp_sum < best_sum*0.8 and bad_point == last_bad_point:
            best_k,best_b = k,b
            last_bad_point,best_sum = bad_point,tmp_sum
        # elif bad_point == 0 :
        #     break

    return best_k,best_b,last_bad_point

def Recur_ransac(beg,edge):
    judge = 0.15
    # unit is cm not m or mm
    thred = 2.0

    tmp = beg
    while beg < edge.shape[0]-10:
        point = edge[beg:beg+4]
        for j in range(1,4):
            if abs(point[j][0] - point[j-1][0]) > 4.0:
                beg = beg + j
                break
        if tmp == beg:
            break
        else:
            tmp = beg
    if beg >= edge.shape[0]-10:
        return edge[beg],edge[beg],beg

    change = 1
    if (abs(edge[beg:beg+4,0] - np.mean(edge[beg:beg+4,0]))\
         <= 0.5).all():
        sum = np.sum(edge[beg:beg+4,0]) 
        i = beg+4
        while  i <= edge.shape[0]:
            if abs(edge[i,1]-edge[i-1,1])>=1.5:
                res_i = i
                break
            tmp = sum*1.0/(i-beg)
            if abs(edge[i,0] - tmp) <= 0.65:
                sum = sum + edge[i,0]
                point = np.append(point,edge[i].reshape(1,2),axis = 0)
                i = i + 1
            else:
                res_i = i
                break
        
        if ((point[0][0]-point[-1][0])**2+(point[0][1]-point[-1][1])\
            **2)**0.5 <= 3:
            change = 1
        else: 
            change = 0
            m = sum*1.0/(i - beg)
            res_beg = np.array([m,edge[beg][1]])
            res_end = np.array([m,edge[res_i-1][1]])
            # edge[beg][0],edge[res_i-1][0] = m,m

    
    if change == 1:
        k,b = new_kb(point)
        m = 0
        out_point = 0

        for i in range(beg+4,edge.shape[0]):
            m = m+1 
            dis = abs(getDis(edge[i][0],edge[i][1],0,b,1,k+b))
            if dis >= 3*thred or abs(edge[i][0]-edge[i-1][0]) > 5.0\
                 or abs(k*edge[i][0]+b-edge[i][1]) > 2.5:
                res_i = i
                break
            elif dis <= thred:
                point = np.append(point,edge[i].reshape(1,2),axis = 0)
            else:
                out_point = out_point + 1
                if out_point == 1:
                    res_point_index = point.size/2
                    res_i = i
                elif out_point > 3:
                    point = point[:res_point_index]
                    break
                point = np.append(point,edge[i].reshape(1,2),axis = 0)
   
            if m%10 == 0:
                k,b,bad_point = ransac_kb(point,k,b)
                if(bad_point*1.0/(i-beg+1) > judge):
                    res_i = i
                    break
        if (point[-1] == edge[-1]).all():
            res_i = edge.shape[0]-1

        k,b,bad_point = ransac_kb(point,k,b)
        res_beg = np.array([edge[beg][0],k*edge[beg][0]+b])
        res_end = np.array([edge[res_i-1][0],k*edge[res_i-1][0]+b])
        # edge[beg][1] = k*edge[beg][0]+b
        # edge[res_i-1][1] = k*edge[res_i-1][0]+b

    if (edge[-1] == point[-1]).all():
        res_i = edge.shape[0]
    
    
    # p_beg,p_end = edge[beg],edge[res_i-1]
    return res_beg,res_end,res_i
            

def ransac(edge,w_min,h,rate):
    rate = 100.0/rate
    ola_edge = np.array([],dtype= float)
    result = np.array([0,0,0,0],dtype= float)
    result = result.reshape(1,4)

    # add actual deep and hight(maybe some trouble)
    for i in edge:
        tmp = [w_min+i[1]*rate,(h-i[0])*rate]
        ola_edge = np.append(ola_edge,tmp,axis = 0)
    ola_edge = ola_edge.reshape((ola_edge.size/2,2))
    # ola_edge[:,1] = -ola_edge[:,1]

    # clear not good edge point
    s,i = ola_edge.shape[0],0
    while i < s-1:
        if ((ola_edge[i+1][0] -ola_edge[i][0])**2+(ola_edge[i+1][1]-ola_edge[i][1])**2)**0.5 >= 30:
            if i != 0 and s-i > 2:
                if ((ola_edge[i+2][0] -ola_edge[i][0])**2+(ola_edge[i+2][1]-ola_edge[i][1])**2)**0.5 <= 10:
                    ola_edge = np.delete(ola_edge,i+1,axis=0)
                    s = s - 1
            elif i == 0:
                if ((ola_edge[i+1][0] -ola_edge[i+2][0])**2+(ola_edge[i+1][1]-ola_edge[i+2][1])**2)**0.5 <= 10:
                    ola_edge = np.delete(ola_edge,i,axis=0)
                    s = s - 1
        i = i + 1

    # polling search lines
    i = 0
    # ola_edge = ola_edge[np.argsort(ola_edge[:,0]),:]
    while i <= ola_edge.shape[0]-10:
        p_beg,p_end,res_i = Recur_ransac(i,ola_edge)
        if ((p_beg[0] -p_end[0])**2+(p_beg[1] -p_end[1])**2)**0.5 <= 3:
            i = i + 1
        elif res_i - i < 10:
            i = i + 1
        else:
            tmp = np.array([p_beg,p_end]).reshape((1,4))
            result = np.append(result,tmp,axis = 0)
            i = res_i
    result = np.delete(result,0,axis = 0)
    return result

def Get_k(l):
    if l[0] - l[2] != 0:
        return (l[1] - l[3])/(l[0] - l[2])
    else:
        return 999999

def Get_len(l):
    return ((l[1] -l[3])**2+(l[0] -l[2])**2)**0.5

def Get_b(l,k):
    if k!= 999999:
        b = l[1] - l[0]*k
    else:
        b = 0
    return b

def line_sort(l):
    if l[2] < l[1]:
        tmp = copy.deepcopy(l[2:])
        l[2:] = l[0:2]
        l[0:2] = tmp
    return l

def Merg_lines(l1,l2,k1,k2):
    l1 = line_sort(l1)
    l2 = line_sort(l2)

    len1 = Get_len(l1)
    len2 = Get_len(l2)
    len = len1 + len2
    
    if len1 > 10*len2:
        k1 = Get_k(l1)
        b1 = Get_b(l1,k1)
        if l1[2] < l2[2]:
            l1[2],l1[3] = l2[2],k1*l2[2]+b1
            return l1
        elif l1[0] > l2[0]:
            l1[0],l1[1] = l2[0],k1*l2[0]+b1
            return l1
    elif len2 > 10*len1:
        k2 = Get_k(l2)
        b2 = Get_b(l2,k2)
        if l2[2] < l1[2]:
            l2[2],l2[3] = l1[2],k2*l1[2]+b2
            return l2
        elif l2[0] > l1[0]:
            l2[0],l2[1] = l1[0],k2*l1[0]+b2
            return l2

    r1 = int(40*len1/len)
    r2 = (40-int(40*len1/len))
    # stride
    s1 = (l1[2] - l1[0])/r1
    s2 = (l2[2] - l2[0])/r2
    if s1 != 0:
        x1 = np.arange(l1[0],l1[2],s1)
    else :
        x1 = l1[2]*np.ones(r1,dtype= float)
    if s2 != 0:
        x2 = np.arange(l2[0],l2[2],s2)
    else :
        x2 = l2[2]*np.ones(r2,dtype= float)

    s1 = (l1[3] - l1[1])/r1
    s2 = (l2[3] - l2[1])/r2
    if s1 != 0:
        y1 = np.arange(l1[1],l1[3],s1)
    else:
        y1 = l1[3]*np.ones(r1,dtype= float)
    if s2 != 0:
        y2 = np.arange(l2[1],l2[3],s2)
    else:
        y2 = l2[3]*np.ones(r2,dtype= float)
    
    if x1.size > r1:
        x1 = x1[:r1]
    if x2.size > r2:
        x2 = x2[:r2]
    if y1.size > r1:
        y1 = y1[:r1]
    if y2.size > r2:
        y2 = y2[:r2]
    x = np.concatenate((x1,x2),axis=0)    
    y = np.concatenate((y1,y2),axis=0)

    l = np.array([0.0,0.0,0.0,0.0])
    point = np.array([x,y]).T
    k,b = new_kb(point)
    l[1],l[0] = k*l1[0]+b,l1[0]
    l[3],l[2] = k*l2[2]+b,l2[2]

    return l

def get_lastline(nl):
    if nl.size == 4:
        l = nl
    elif nl.shape[0] > 1:
        l = nl[-1]
    else:
        l = np.array([])
    return l
    

def add_lines(nl,l):
    if nl.size != 0:
        nl = np.row_stack((nl,l))
    else:
        nl = l
    return nl

def delete_lines(nl):
    if nl.size == 0:
        return nl
    else:
        if nl.size == 4:
            nl = np.array([])
        else:
            nl = np.delete(nl,-1,axis=0)
        return nl

# type: 0 is normal/slope, 1 is upstair,2 is downstair,3 is obstacle
# don't have error norm judge
def connect_ransac(lines,type_road = 0):
    if type_road == 0:
        # new_lines = lines
        if lines.size == 4:
            new_lines = lines
            new_lines = new_lines.reshape(new_lines.size/4,4)
            return new_lines    
        else:
            new_lines = np.array([])

        i = 0
        k1 = Get_k(lines[i])
        len1 = Get_len(lines[i])
        while i < lines.shape[0]-1:
            k2 = Get_k(lines[i+1])
            len2 = Get_len(lines[i+1])
            tan1 = abs(math.atan(k1) - math.atan(k2))
            if tan1 < 0.05 or tan1 <0.6 and (len2 >3*len1 or len1 > 3*len2):
                # new_lines = delete_lines(new_lines)
                lines[i+1] = Merg_lines(lines[i],lines[i+1],k1,k2)
                l = get_lastline(new_lines)    
                if l.size != 1 and l.size != 0:
                    tmp_x,tmp_y = line_intersection(l,lines[i+1])
                    l[2],l[3] = tmp_x,tmp_y
                    lines[i+1][0],lines[i+1][1] = tmp_x,tmp_y
                    new_lines = delete_lines(new_lines)
                    new_lines = add_lines(new_lines,l)
                elif l.size == 0:
                    new_lines = add_lines(new_lines,lines[i+1])
            else :
                tmp_x,tmp_y = line_intersection(lines[i],lines[i+1])
                if tmp_x > lines[i][0]:
                    lines[i][2],lines[i][3] = tmp_x,tmp_y
                    lines[i+1][0],lines[i+1][1] = tmp_x,tmp_y
                    new_lines = add_lines(new_lines,lines[i])
                elif len2 > 3*len1 and k2 < 0.179:
                    lines[i][3] = k2*(lines[i][2]-lines[i][0])+lines[i][1]
                    lines[i+1] = Merg_lines(lines[i],lines[i+1],k2,k2)
                    l = get_lastline(new_lines)    
                    if l.size != 1 and l.size != 0:
                        tmp_x,tmp_y = line_intersection(l,lines[i+1])
                        l[2],l[3] = tmp_x,tmp_y
                        lines[i+1][0],lines[i+1][1] = tmp_x,tmp_y
                        new_lines = delete_lines(new_lines)
                        new_lines = add_lines(new_lines,l)
                    elif l.size == 0:
                        new_lines = add_lines(new_lines,lines[i+1])
                elif len1 > 3*len2 and k1 < 0.179:
                    lines[i+1][3] = k1*(lines[i+1][2]-lines[i+1][0])+lines[i+1][1]
                    lines[i+1] = Merg_lines(lines[i],lines[i+1],k1,k1)
                    l = get_lastline(new_lines)    
                    if l.size != 1 and l.size != 0:
                        tmp_x,tmp_y = line_intersection(l,lines[i+1])
                        l[2],l[3] = tmp_x,tmp_y
                        lines[i+1][0],lines[i+1][1] = tmp_x,tmp_y
                        new_lines = delete_lines(new_lines)
                        new_lines = add_lines(new_lines,l)
                    elif l.size == 0:
                        new_lines = add_lines(new_lines,lines[i+1])
                else:
                    tmp_x,tmp_y = lines[i][2],k2*(lines[i][2]-lines[i+1][0])+lines[i+1][1]
                    l = np.array([lines[i][2],lines[i][3],tmp_x,tmp_y])
                    lines[i+1][0],lines[i+1][1] = tmp_x,tmp_y
                    new_lines = add_lines(new_lines,lines[i])
                    new_lines = add_lines(new_lines,l)
                

            if i+1 == lines.shape[0]-1:
                new_lines = np.row_stack((new_lines,lines[i+1]))
                break
            k1,len1 = Get_k(lines[i+1]),Get_len(lines[i+1])
            i = i+1
    elif type_road == 1:
        new_lines = np.array([])
        i ,j= 0,0
        while i < lines.shape[0]:
            k = Get_k(lines[i])
            len = Get_len(lines[i])
            if k < 0.25 and k > -0.25 and len >= 4:
                if i < lines.shape[0]-1:
                    j = i+1
                else:
                    break
                while 1:
                    k1 = Get_k(lines[j])
                    len1 = Get_len(lines[j])
                    tan1 = abs(math.atan(k1) - math.atan(k))
                    # if tan1 < 0.175:
                    # if tan1 < 0.2:
                    if k1 < 0.25 and k1 > -0.25 and len1 >= 4:
                        if abs(k1) < abs(k) and abs(lines[i][1]- lines[j][1]) < 3:
                            i,k,len = j,k1,len1
                        elif abs(lines[i][3]- lines[j][3]) > 5:
                            if j != lines.shape[0]-1:
                                for jj in range(j,lines.shape[0]):
                                    k2 = Get_k(lines[jj])
                                    len2 = Get_len(lines[jj])
                                    tan2 = abs(math.atan(k2) - math.atan(k))
                                    if tan2 < tan1 and len2 >= 4 and\
                                         abs(lines[j][3]- lines[jj][3]) < 3:
                                        j ,k1= jj,k2
                                        len1 = len2
                                    elif abs(lines[j][3]- lines[jj][3]) > 4:
                                        break
                            break       
                        
                    if j != lines.shape[0]-1:                    
                        j = j +1           
                    else:
                        j = None
                        break
                
                if j != None:
                    tmp_h,tmp_k1,tmp_k = 0,0,i+1
                    if i+1 < j:
                        for k in range(i+1,j):
                            k1 = Get_k(lines[k])
                            len1 = Get_len(lines[k])
                            if abs(k1) > abs(tmp_k1) and len1>= 3:
                                tmp_h = len1
                                tmp_k1 = k1
                                tmp_k = k
                                if tmp_k == 999999:
                                    break
                        tmp_x,tmp_y = line_intersection(lines[i],lines[tmp_k])
                        lines[i][2],lines[i][3] = tmp_x,tmp_y
                        lines[tmp_k][0],lines[tmp_k][1] = tmp_x,tmp_y
                        new_lines = add_lines(new_lines,lines[i])
                        tmp_x,tmp_y = line_intersection(lines[tmp_k],lines[j])
                        lines[tmp_k][2],lines[tmp_k][3] = tmp_x,tmp_y
                        lines[j][0],lines[j][1] = tmp_x,tmp_y
        
                        new_lines = np.row_stack((new_lines,lines[tmp_k]))
                        # new_lines = np.row_stack((new_lines,lines[j]))
                        tmp_lines = np.array([])
                        tmp_k,tmp_h = 0,0
                        if j+6 >= lines.shape[0]:
                            end = lines.shape[0]
                        else:
                            end = j+6 
                        for k in range(j+1,end):
                            k1 = Get_k(lines[k])
                            len1 = Get_len(lines[k])
                            if abs(k1) > abs(tmp_k) and abs(len1) > 4 and abs(lines[k][0]-lines[j][2]) < 6:
                                tmp_k = k1
                                tmp_lines = lines[k]
                                if k1 == 999999:
                                    break
                        if tmp_lines.size == 0:
                            tmp_lines = np.array([lines[j][2],lines[j][3],\
                                lines[j][2],lines[j][3]+5])
                        tmp_x,tmp_y = line_intersection(tmp_lines,lines[j])
                        if tmp_lines[3] >tmp_lines[1]: 
                            tmp_lines[0],tmp_lines[1] = tmp_x,tmp_y
                        else :
                            tmp_lines[2],tmp_lines[3] = tmp_x,tmp_y
                        lines[j][2],lines[j][3] = tmp_x,tmp_y

                        new_lines = np.row_stack((new_lines,lines[j]))
                        new_lines = np.row_stack((new_lines,tmp_lines))
                        # new_lines = new_lines.append(new_lines,lines[j])
                    i = j-1
                    break
            i = i+1  
    elif type_road == 2:
        new_lines = np.array([])
        i ,j= 0,0
        while i < lines.shape[0]:
            k = Get_k(lines[i])
            len = Get_len(lines[i])
            if k < 0.25 and k > -0.25 and len >= 4:
                if i < lines.shape[0]-2:
                    j = i+1
                else:
                    break
                while 1:
                    k1 = Get_k(lines[j])
                    len1 = Get_len(lines[j])
                    tan1 = abs(math.atan(k1) - math.atan(k))
                    # if tan1 < 0.175:
                    if tan1 < 0.2:
                        if abs(lines[i][3]-lines[j][1]) > 5 and j != lines.shape[0]-1:
                            k2 = Get_k(lines[j+1])
                            len2 = Get_len(lines[j+1])
                            tan2 = abs(math.atan(k2) - math.atan(k))
                            if tan2 < tan1 and len2 > len1:
                                j ,k1= j+1,k2
                                len1 = len2
                                break
                            else:
                                break
                        elif abs(lines[i][3]-lines[j][1]) < 3 and abs(lines[i][3]-lines[j][3]) <5\
                            and abs(lines[i][2]-lines[j][0]) < 2 and j < i+3:
                            lines[i] = Merg_lines(lines[i],lines[j],k,k1)
                            k = Get_k(lines[i])
                        elif abs(k1) < abs(k):
                            i,k = j,k1
                        else:
                            break
                    if j != lines.shape[0]-1:                    
                        j = j +1           
                    else:
                        j = None
                        break
                if j != None:
                    tmp_lines = np.zeros(4)

                    k2 = Get_k(lines[j])
                    b2 = Get_b(lines[j],k2)
                    lines[j][0],lines[j][1] = lines[i][2],k2*lines[i][2]+b2
                    tmp_lines[0],tmp_lines[1],tmp_lines[2],tmp_lines[3]=\
                        lines[i][2],lines[i][3],lines[j][0],lines[j][1]
                    new_lines = add_lines(new_lines,lines[i])
                    new_lines = np.row_stack((new_lines,tmp_lines))

                    if j+1 != lines.shape[0]-1:
                        jj = j+1
                        while jj < lines.shape[0] and jj < j+4:
                            k1 = Get_k(lines[jj])
                            tan1 = abs(math.atan(k1) - math.atan(k2))
                            if tan1 > 0.175:
                                lines[j][2],lines[j][3] = lines[jj][0],k2*lines[jj][0]+b2
                                break
                            elif abs(lines[jj][1]-lines[jj][3]) > 3:
                                lines[j][2],lines[j][3] = lines[jj][0],k2*lines[jj][0]+b2
                                break
                            elif abs(k1) < abs(k2):
                                lines[j][2], lines[j][3] = lines[jj][2],\
                                    k1*(lines[jj][2]-lines[j][0])+lines[j][1]
                                k2 = k1 
                            jj = jj + 1
                        
                    tmp_lines[0],tmp_lines[1],tmp_lines[2],tmp_lines[3]=\
                            lines[j][2],lines[j][3],lines[j][2],lines[j][3]-5
                    new_lines = np.row_stack((new_lines,tmp_lines))
                    new_lines = np.row_stack((new_lines,lines[j]))

                    break
            
            i = i+1  
    
    elif type_road == 3:
        new_lines = np.array([])
        ind = np.array([])
        i = 0
        while i < lines.shape[0]:
            if abs(lines[i][1]) <= 20 and abs(lines[i][3]) <= 20:
                if abs(lines[i][3]-lines[i][1]) > 4:
                    new_lines = add_lines(new_lines,lines[i])
                    if ind.size != 0:
                        ind = np.insert(ind,ind.size,i)
                    else:
                        ind = np.array([i])
                elif abs(Get_k(lines[i])) < 0.2:
                    new_lines = add_lines(new_lines,lines[i])
                    if ind.size != 0:
                        ind = np.insert(ind,ind.size,i)
                    else:
                        ind = np.array([i])
            i = i + 1
        if ind.size == 0:
            return new_lines 
        elif new_lines.size != 4:
            i = 0
            k1 = Get_k(new_lines[i])
            len1 = Get_len(new_lines[i])
            while i < new_lines.shape[0]-1:
                k2 = Get_k(new_lines[i+1])
                len2 = Get_len(new_lines[i+1])
                tan1 = abs(math.atan(k1) - math.atan(k2))
                if abs(k1) < 0.179 and (tan1 < 0.05 or tan1 <0.6 and len2 >3*len1):
                    # new_lines = delete_lines(new_lines)
                    new_lines[i] = Merg_lines(new_lines[i],new_lines[i+1],k1,k2)
                    new_lines = np.delete(new_lines,i+1,axis=0)
                    ind = np.delete(ind,i+1)
                    continue
                elif abs(new_lines[i][2] - new_lines[i+1][0]) < 3:
                    tmp_x,tmp_y = line_intersection(new_lines[i],new_lines[i+1])
                    new_lines[i][2],new_lines[i][3] = tmp_x,tmp_y
                    new_lines[i+1][0],new_lines[i+1][1] = tmp_x,tmp_y
                    i = i + 1
                    k1 = Get_k(new_lines[i])
                    len1 = Get_len(new_lines[i])
                    continue
                else:
                    i = i + 1
                    k1 = Get_k(new_lines[i])
                    len1 = Get_len(new_lines[i])

        if ind.size == 1:
            new_lines = new_lines.reshape(4,)
            fir_x,h = new_lines[2],0
            for i in range(ind[0],lines.shape[0]):
                if lines[i][0] <  fir_x and lines[i][2] > lines[i][0]\
                 and abs(lines[i][1]) > 20:
                    fir_x = lines[i][0]
                elif lines[i][2] <  fir_x and lines[i][0] > lines[i][2]\
                 and abs(lines[i][3]) > 20:
                    fir_x = lines[i][2]
                if lines[i][1] > h and lines[i][1] > lines[i][3]:
                    h= lines[i][1]
                elif lines[i][3] > h:
                    h = lines[i][3]
            l = np.array([fir_x,new_lines[3],fir_x,h])
            new_lines[2] = fir_x
            new_lines = add_lines(new_lines,l)
        # lose code





        # fir_x,sec_X = lines[0][0],lines[0][2]
        if ind[0] != 0:
            fir_x,sec_X = lines[0][0],lines[ind[0]][0]
            h = 0
            for i in range(0,ind[0]):
                if abs(lines[i][1]) > abs(lines[i][3]): 
                    tmp_h = lines[i][1]
                else:
                    tmp_h = lines[i][3]
                # tmp_len = abs(lines[i][2]-lines[i][0])
                if abs(tmp_h) > abs(h):
                    h = tmp_h
            if h > 25:
                sec_X = fir_x

            l = np.array([fir_x,0,sec_X,h])
            new_lines = add_lines(new_lines,l)


        

    new_lines = new_lines.reshape(new_lines.size/4,4)
    return new_lines     

def point_to_pixel(p,Ri,Rba):
    p = np.dot(Rba,p)
    p = np.dot(Ri,p)
    p = p/p[2]
    p = p.astype(np.int16)
    if p[0] < 0:
        p[0] = 0
    elif p[0] > 640:
        p[0] = 640
    if p[1] < 0:
        p[1] = 0
    elif p[1] > 480:
        p[1] = 480
    return p


def draw_line(img,edge,Ri,Rba):
    if edge.size <= 5:
        return img 
    else:
        ret = img
    # lr = -0.1
    color = [0,0,255]
    # edge[:,1] = -0.55
    fitting_point = np.array([])
    
    for i in range(0,edge.shape[0]):
        # 1.17 is pre-define ,if use in robot that high will change
        pixel1 = np.array([edge[i][1],body_high-edge[i][2]/100.0,edge[i][0]/100.0])

        # good children don't use it
        # pixel2 = np.array([lr,0.97-edge[i-1][4]/100.0,edge[i-1][0]/100.0])
        pixel2 = np.array([edge[i][3],body_high-edge[i][4]/100.0,edge[i][0]/100.0])
        
        # pixel3 = np.array([edge[i][1],1.07-edge[i][2]/100.0,edge[i][0]/100.0])
        # # pixel4 = np.array([lr,0.97-edge[i][4]/100.0,edge[i][0]/100.0])
        # pixel4 = np.array([edge[i][3],1.07-edge[i][4]/100.0,edge[i][0]/100.0])

        # use inline fuction may more effective
        pixel1 = point_to_pixel(pixel1,Ri,Rba)
        pixel2 = point_to_pixel(pixel2,Ri,Rba)
        # pixel3 = point_to_pixel(pixel3,Ri,Rba)
        # pixel4 = point_to_pixel(pixel4,Ri,Rba)

        fitting_point = add_lines(fitting_point,pixel1[:2])
        fitting_point = add_lines(fitting_point,pixel2[:2])
        # fitting_point = add_lines(fitting_point,pixel3[:2])
        # fitting_point = add_lines(fitting_point,pixel4[:2])
        
        # ret = cv2.line(img,tuple(pixel1[:2]),tuple(pixel3[:2]),color,2)
        # ret = cv2.line(img,tuple(pixel2[:2]),tuple(pixel4[:2]),color,2)
    
    fitting_point = fitting_point.reshape(fitting_point.size/4,4)
    x = fitting_point[:,0]
    y = fitting_point[:,1]
    if np.unique(x).size > 1 or np.unique(y).size > 1:
        z1 = np.polyfit(x, y, 1)
        k,b = z1[0],z1[1]

        if np.mean(edge[:,2]) < 10:
            color = [0,255,0]
        else:
            color = [0,0,255]
        ret = cv2.line(img,tuple([x[0],int(k*x[0]+b)]),tuple([x[-1],int(k*x[-1]+b)]),color,2)
    
    x = fitting_point[:,2]
    y = fitting_point[:,3]
    if np.unique(x).size > 1 or np.unique(y).size > 1:
        z1 = np.polyfit(x, y, 1)
        k,b = z1[0],z1[1]

        if np.mean(edge[:,4]) < 10:
            color = [0,255,0]
        else:
            color = [0,0,255]
        ret = cv2.line(img,tuple([x[0],int(k*x[0]+b)]),tuple([x[-1],int(k*x[-1]+b)]),color,2)
    
    return ret

def Get_Obstacle_para_simplfiy(ransac):
    # status 0 is road,1 is obstacle
    status = 0
    for r in ransac:
        if status == 0 and abs(r[3]) < 15:
            res_distance = r[2]
        elif status == 0 and abs(r[3]) >= 15:
            if Get_len(r) < 5 and abs(ransac[ransac.index(r)+1][3]) < 15:
                continue
            else:
                status,res_high = 1,r[3]
                if r[1] < 15:
                    res_distance = r[0]
        elif status == 1 and abs(r[1]) > res_high and abs(r[3]) > res_high:
           if  abs(r[1]) > abs(r[3]):
               res_high = r[1]
            else:
                res_high = r[3]
    return res_distance,res_high


def draw_walkarea_withlabel(point,ransac,img,Ri,Rba,w_min,scene):
    point[1],point[2] = point[1]*100,point[2]*100
    # Height difference lower limits
    ll = 1.2
    
    for i in range(0,ransac.shape[0]):
        k = Get_k(ransac[i])
        if Get_len(ransac[i]) > 5.0 and \
            abs(k) < 1.0 and abs(ransac[i][1]) < 20:
            # abs(k) < 10 and abs(ransac[i][1]) < 28:
            
            if ransac[i][2] < ransac[i][0]:
                tmp_x,tmp_y =  ransac[i][0],ransac[i][1]
                ransac[i][0],ransac[i][1] = ransac[i][2],ransac[i][3]
                ransac[i][2],ransac[i][3] = tmp_x,tmp_y

            beg = ransac[i][0]+w_min*100
            b = Get_b(ransac[i],k)
            h = k*(beg-w_min*100)+b
            edge = np.array([])
            while beg < ransac[i][2]+w_min*100:
                im =  (point[2] > beg-2) & (point[2] < beg+2)
                tmp_p1 = point[:,im]
                im = (tmp_p1[1] > h-10) & (tmp_p1[1] < h+20)
                tmp_p2 = tmp_p1[:,im]
                if tmp_p2.size == 0:
                    left,right = np.array([-0.3,h]),np.array([0.3,h])
                else:
                    # left,right = tmp_p2[0].min(),tmp_p2[0].max()
                    im = (tmp_p1[0] < -0.3) & (tmp_p1[0] > tmp_p2[0].min())
                    tmp_p3 = tmp_p1[:,im]
                    if tmp_p3.size > 4:
                        tmp_p3 = tmp_p3[:,np.argsort(tmp_p3[0,:])]
                        tmp_p3 = tmp_p3[:,::-1]
                        error_num,h_sum = 0,[]
                        for j in range(0,tmp_p3.shape[1]):
                            if j != 0 and abs(tmp_p3[1][j] - np.mean(h_sum)) > ll:
                                error_num = error_num + 1
                                if error_num > 0.2*tmp_p3.shape[1]:
                                    if j + 1 - error_num != 0:
                                        h_mean = np.mean(h_sum)
                                    else:
                                        h_mean = h
                                    left = np.array([tmp_p3[0][j],h_mean])
                                    break
                            else:
                                left = np.array([tmp_p3[0][j],tmp_p3[1][j]])
                                h_sum.append(tmp_p3[1][j])
                    else:
                        left = np.array([-0.3,h])
                    
                    im = (tmp_p1[0] > 0.3) & (tmp_p1[0] < tmp_p2[0].max())
                    tmp_p3 = tmp_p1[:,im]
                    if tmp_p3.size > 4:
                        tmp_p3 = tmp_p3[:,np.argsort(tmp_p3[0,:])]
                        error_num,h_sum = 0,[]
                        for j in range(0,tmp_p3.shape[1]):
                            if j!= 0 and abs(tmp_p3[1][j] - np.mean(h_sum)) > ll:
                                error_num = error_num + 1
                                if error_num > 0.2*tmp_p3.shape[1]:
                                    if j + 1 - error_num != 0:
                                        h_mean = np.mean(h_sum)
                                    else:
                                        h_mean = h
                                    right = np.array([tmp_p3[0][j],h_mean])
                                    break
                            else:
                                right = np.array([tmp_p3[0][j],tmp_p3[1][j]])
                                h_sum.append(tmp_p3[1][j])
                    else:
                        right = np.array([0.3,h])
                
                edge = add_lines(edge,np.concatenate((np.array([beg]),left,right),axis=0))
                beg = beg + 3
                h = k*(beg-w_min*100)+b
            if edge.size != 0:
                img = draw_line(img,edge,Ri,Rba)
    
    font=cv2.FONT_HERSHEY_SIMPLEX

    txt = "Result:"+scene
    cv2.putText(img,txt,(10,50), font, 1,(127,255,0),2)

    if scene == "obstacle":
        dis,high = Get_Obstacle_para_simplfiy(ransac)
        txt = "distance:"+str(dis/100)+"m"
        cv2.putText(img,txt,(10,80), font, 1,(127,255,0),2)
        txt = "obstacle_high:"+str(high)+"cm"
        cv2.putText(img,txt,(10,110), font, 1,(127,255,0),2)
    elif scene == "outdoor":
        txt = "road_width:1.17m"
        cv2.putText(img,txt,(10,80), font, 1,(127,255,0),2)
        txt = "slope degree:-7.237"
        cv2.putText(img,txt,(10,110), font, 1,(127,255,0),2)
        txt = "slope distance:1.66m"
        cv2.putText(img,txt,(10,140), font, 1,(127,255,0),2)

    # txt = "road_width:1.17m"
    # txt = "max_height:52.7cm"
    # cv2.putText(img,txt,(10,80), font, 1,(127,255,0),2)

    
    # txt = "slope degree:-7.237"
    # txt = "step_height:16.5cm"
    # cv2.putText(img,txt,(10,110), font, 1,(127,255,0),2)

    # txt = "slope distance:1.66m"
    # txt = "step_width:22.9cm"
    # cv2.putText(img,txt,(10,140), font, 1,(127,255,0),2)
    
    return img 



        
                    