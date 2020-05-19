import numpy as np
import math

def line_intersection(line1, line2):
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
    point = point.reshape((point.size/2,2))
    x = point[:,0]
    y = point[:,1]
    z1 = np.polyfit(x, y, 1)
    k = z1[0]
    b = z1[1]
    return k,b


def Recur_ransac(beg,edge):
    judge = 0.05
    # unit is cm not m or mm
    meter = 1.0
    point = edge[beg:beg+4]

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
        m = sum*1.0/(i - beg)
        edge[beg][0],edge[res_i-1][0] = m,m
        
    else:
        k,b = new_kb(point)
        m = 0
        for i in range(beg+4,edge.shape[0]):
            m = m+1
            if(abs(getDis(edge[i][0],edge[i][1],0,b,1,k+b)) <= meter):
                point = np.append(point,edge[i].reshape(1,2),axis = 0)
            else:
                res_i = i
                break
            if m%5 == 0:
                k,b = new_kb(point)
                bad_point = 0
                for j in range(beg,i+1):
                    dis = abs(getDis(edge[j][0],edge[j][1],0,b,1,k+b))
                    if(dis > meter):
                        bad_point = bad_point +1
                if(bad_point*1.0/(i-beg+1) > judge):
                    res_i = i
                    break
        k,b = new_kb(point)
        edge[beg][1] = k*edge[beg][0]+b
        edge[i-1][1] = k*edge[i-1][0]+b

    if (edge[-1] == point[-1]).all():
        res_i = edge.shape[0]
    
    
    p_beg,p_end = edge[beg],edge[res_i-1]
    return p_beg,p_end,res_i
            

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
            i = i+1
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

def Merg_lines(l1,l2,k1,k2):
    len1 = Get_len(l1)
    len2 = Get_len(l2)
    len = len1 + len2

    r1 = int(40*len1/len)
    r2 = (40-int(40*len1/len))
    # stride
    s1 = (l1[2] - l1[0])/r1
    s2 = (l2[2] - l2[0])/r2
    x1 = np.arange(l1[0],l1[2],s1)
    x2 = np.arange(l2[0],l2[2],s2)
    s1 = (l1[3] - l1[1])/r1
    s2 = (l2[3] - l2[1])/r2
    y1 = np.arange(l1[1],l1[3],s1)
    y2 = np.arange(l2[1],l2[3],s2)
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

# type: 0 is normal/slope, 1 is upstair,2 is downstair
def connect_ransac(lines,type_road = 0):
    if type_road == 0:
        new_lines = lines
        new_lines = np.array([])
        i = 0
        while i < lines.shape[0]-1:
            k1 = Get_k(lines[i])
            k2 = Get_k(lines[i+1])
            tan1 = abs(math.atan(k1) - math.atan(k2))
            if tan1 > 0.05:
                tmp_x,tmp_y = line_intersection(lines[i],lines[i+1])
                lines[i][2],lines[i][3] = tmp_x,tmp_y
                lines[i+1][0],lines[i+1][1] = tmp_x,tmp_y
                new_lines = add_lines(new_lines,lines[i])
                if i+1 == lines.shape[0]-1:
                    new_lines = np.row_stack((new_lines,lines[i+1]))
                    break
            else:
                new_lines = delete_lines(new_lines)
                l = Merg_lines(lines[i],lines[i+1],k1,k2)
                lines[i+1] = l 
                new_lines = add_lines(new_lines,lines[i+1])
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
                    if tan1 < 0.2:
                        if j != lines.shape[0]-1:
                            k2 = Get_k(lines[j+1])
                            len2 = Get_len(lines[j+1])
                            tan2 = abs(math.atan(k2) - math.atan(k))
                            if tan2 < tan1 and len2 > len1:
                                j ,k1= j+1,k2
                                len1 = len2
                                break
                            else:
                                break
                        else:
                            break
                    if j != lines.shape[0]-1:                    
                        j = j +1           
                    else:
                        j = None
                        break
                if j != None:
                    tmp_h,tmp_x,tmp_k = 0,0,i+1
                    if i+1 < j:
                        for k in range(i+1,j):
                            k1 = Get_k(lines[k])
                            len1 = Get_len(lines[k])
                            if abs(k1) > abs(tmp_k) and len1> tmp_h:
                                tmp_h = len1
                                tmp_x = k1
                                tmp_k = k
                        tmp_x,tmp_y = line_intersection(lines[i],lines[tmp_k])
                        lines[i][2],lines[i][3] = tmp_x,tmp_y
                        lines[tmp_k][0],lines[tmp_k][1] = tmp_x,tmp_y
                        new_lines = add_lines(new_lines,lines[i])
                        tmp_x,tmp_y = line_intersection(lines[tmp_k],lines[j])
                        lines[tmp_k][2],lines[tmp_k][3] = tmp_x,tmp_y
                        lines[j][0],lines[j][1] = tmp_x,tmp_y
        
                        new_lines = np.row_stack((new_lines,lines[tmp_k]))
                        # new_lines = np.row_stack((new_lines,lines[j]))
                        
                        tmp_k,tmp_h = 0,0
                        for k in range(j+1,j+4):
                            k1 = Get_k(lines[k])
                            len1 = Get_len(lines[k])
                            if abs(k1) > abs(tmp_k) and abs(len1) > 5:
                                tmp_k = k1
                                tmp_lines = lines[k]
                                if k1 == 999999:
                                    break
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
            if k < 0.25 and k > -0.25 and len >= 10:
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
                        if j != lines.shape[0]-1:
                            k2 = Get_k(lines[j+1])
                            len2 = Get_len(lines[j+1])
                            tan2 = abs(math.atan(k2) - math.atan(k))
                            if tan2 < tan1 and len2 > len1:
                                j ,k1= j+1,k2
                                len1 = len2
                                break
                            else:
                                break
                        else:
                            break
                    if j != lines.shape[0]-1:                    
                        j = j +1           
                    else:
                        j = None
                        break
                if j != None:
                    tmp_h,tmp_k = 0,i+1
                    if i+1 < j:
                        for k in range(i+1,j):
                            k1 = Get_k(lines[k])
                            len1 = Get_len(lines[k])
                            if len1 > tmp_h:
                                tmp_h = len1
                                tmp_k = k
                        tmp_x,tmp_y = line_intersection(lines[i],lines[k])
                        lines[i][2],lines[i][3] = tmp_x,tmp_y
                        lines[k][0],lines[k][1] = tmp_x,tmp_y
                        # new_lines = add_lines(new_lines,lines[i])
                        tmp_x,tmp_y = line_intersection(lines[k],lines[j])
                        lines[k][2],lines[k][3] = tmp_x,tmp_y
                        lines[j][0],lines[j][1] = tmp_x,tmp_y
                        
                        tmp_lines = np.zeros(4)
                        k1 = Get_k(lines[i])
                        if k1 != 0:
                            k1 = 1.0/k1
                            b1 = lines[i][3] - k1*lines[i][2]
                            tmp_lines[0],tmp_lines[1],tmp_lines[2],tmp_lines[3]=\
                                lines[i][2],lines[i][3],1.0,k1+b1
                            tmp_x,tmp_y = line_intersection(tmp_lines,lines[j])
                            lines[j][0],lines[j][1] = tmp_x,tmp_y
                            tmp_lines[2],tmp_lines[3] = tmp_x,tmp_y
                        else:
                            k2 = Get_k(lines[j])
                            b2 = Get_b(lines[j],k2)
                            lines[j][0],lines[j][1] = lines[i][2],k2*lines[i][2]+b2
                            tmp_lines[0],tmp_lines[1],tmp_lines[2],tmp_lines[3]=\
                                lines[i][2],lines[i][3],lines[j][0],lines[j][1]
                        new_lines = add_lines(new_lines,lines[i])
                        new_lines = np.row_stack((new_lines,lines[j]))
                        new_lines = np.row_stack((new_lines,tmp_lines))

                    elif i+1 == j:
                        k2 = Get_k(lines[j])
                        b2 = Get_b(lines[j],k2)
                        lines[j][0],lines[j][1] = lines[i][2],k2*lines[i][2]+b2
                        tmp_lines[0],tmp_lines[1],tmp_lines[2],tmp_lines[3]=\
                            lines[i][2],lines[i][3],lines[j][0],lines[j][1]
                        new_lines = add_lines(new_lines,lines[i])
                        new_lines = np.row_stack((new_lines,lines[j]))
                        new_lines = np.row_stack((new_lines,tmp_lines))
                    break
            
            i = i+1  


    new_lines = new_lines.reshape(new_lines.size/4,4)
    return new_lines     


                    