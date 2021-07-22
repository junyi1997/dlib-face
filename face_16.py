#!/usr/bin/env python
count=[0]  
count_bef=0
count_error=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
count_success=[]
existing_faces_cnt=0
import time 
import cv2
import numpy as np
import dlib
import time
import math
import os
# import sys
# sys.setrecursionlimit(100000)
import features_extraction_to_csv as pic_features
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
POINTS_NUM_LANDMARK = 68

# 获取最大的人脸
def _largest_face(dets):
    if len(dets) == 1:
        return 0

    face_areas = [ (det.right()-det.left())*(det.bottom()-det.top()) for det in dets]

    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area :
            largest_index = index
            largest_area = face_areas[index]

    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

    return largest_index

# 从dlib的检测结果抽取姿态估计需要的点坐标
def get_image_points_from_landmark_shape(landmark_shape):
    if landmark_shape.num_parts != POINTS_NUM_LANDMARK:
        print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
        return -1, None
    
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (landmark_shape.part(30).x, landmark_shape.part(30).y),     # Nose tip
                                (landmark_shape.part(8).x, landmark_shape.part(8).y),     # Chin
                                (landmark_shape.part(36).x, landmark_shape.part(36).y),     # Left eye left corner
                                (landmark_shape.part(45).x, landmark_shape.part(45).y),     # Right eye right corne
                                (landmark_shape.part(48).x, landmark_shape.part(48).y),     # Left Mouth corner
                                (landmark_shape.part(54).x, landmark_shape.part(54).y)      # Right mouth corner
                            ], dtype="double")

    return 0, image_points
    
# 用dlib检测关键点，返回姿态估计需要的几个点坐标
def get_image_points(img):
                            
    #gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 图片调整为灰色
    dets = detector( img, 0 )

    if 0 == len( dets ):
        # print( "ERROR: found no face" )
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]

    landmark_shape = predictor(img, face_rectangle)

    return get_image_points_from_landmark_shape(landmark_shape)


# 获取旋转向量和平移向量                        
def get_pose_estimation(img_size, image_points ):
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])
     
    # Camera internals
     
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
     
    # print("Camera Matrix :{}".format(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )
 
    # print("Rotation Vector:\n {}".format(rotation_vector))
    # print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    
    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    # print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)
    
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    
    # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    
	# 单位转换：将弧度转换为度
    Y = int((pitch/math.pi)*180)
    X = int((yaw/math.pi)*180)
    Z = int((roll/math.pi)*180)
    
    return 0, Y, X, Z

def get_pose_estimation_in_euler_angle(landmark_shape, im_szie):
    try:
        ret, image_points = get_image_points_from_landmark_shape(landmark_shape)
        if ret != 0:
            # print('get_image_points failed')
            return -1, None, None, None
    
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie, image_points)
        if ret != True:
            # print('get_pose_estimation failed')
            return -1, None, None, None
    
        ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        if ret != 0:
            # print('get_euler_angle failed')
            return -1, None, None, None

        euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
        print(euler_angle_str)
        return 0, pitch, yaw, roll
    
    except Exception as e:
        # print('get_pose_estimation_in_euler_angle exception:{}'.format(e))
        return -1, None, None, None
def color_change(number_i,color_i):
    global count
    color1=(0, 0, 255)
    color2=(0, 0, 255)
    color3=(0, 0, 255)
    color4=(0, 0, 255)
    color5=(0, 0, 255)
    color6=(0, 0, 255)
    color7=(0, 0, 255)
    color8=(0, 0, 255)
    color9=(0, 0, 255)
    color10=(0, 0, 255)
    color11=(0, 0, 255)
    color12=(0, 0, 255)
    color13=(0, 0, 255)
    color14=(0, 0, 255)
    color15=(0, 0, 255)
    color16=(0, 0, 255)
    
    if color_i == "red":
        cv2.putText( im, "1", (200, 280), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (0, 135), (477, 337), (0, 0, 255), 2)
        cv2.putText( im, "2", (677, 280), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (477, 135), (954, 337), (0, 0, 255), 2)
        cv2.putText( im, "3", (1154, 280), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (954, 135), (1431, 337), (0, 0, 255), 2)
        cv2.putText( im, "4", (1631, 280), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (1431, 135), (1908, 337), (0, 0, 255), 2)
        cv2.putText( im, "5", (200, 482), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (0, 337), (477, 539), (0, 0, 255), 2)
        cv2.putText( im, "6", (677, 482), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (477, 337), (954, 539), (0, 0, 255), 2)
        cv2.putText( im, "7", (1154, 482), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (954, 337), (1431, 539), (0, 0, 255), 2)
        cv2.putText( im, "8", (1631, 482), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (1431, 337), (1908, 539), (0, 0, 255), 2)
        cv2.putText( im, "9", (200, 684), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (0, 539), (477, 741), (0, 0, 255), 2)
        cv2.putText( im, "10", (630, 684), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (477, 539), (954, 741), (0, 0, 255), 2)
        cv2.putText( im, "11", (1107, 684), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (954, 539), (1431, 741), (0, 0, 255), 2)
        cv2.putText( im, "12", (1584, 684), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (1431, 539), (1908, 741), (0, 0, 255), 2)
        cv2.putText( im, "13", (153, 886), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (0, 741), (477, 943), (0, 0, 255), 2)
        cv2.putText( im, "14", (630, 886), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (477, 741), (954, 943), (0, 0, 255), 2)
        cv2.putText( im, "15", (1107, 886), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (954, 741), (1431, 943), (0, 0, 255), 2)
        cv2.putText( im, "16", (1584, 886), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (1431, 741), (1908, 943), (0, 0, 255), 2)
    else:
        if 1 in count:
            color1=(0, 255, 0)
        if 2 in count:
            color2=(0, 255, 0)
        if 3 in count:
            color3=(0, 255, 0)
        if 4 in count:
            color4=(0, 255, 0)
        if 5 in count:
            color5=(0, 255, 0)
        if 6 in count:
            color6=(0, 255, 0)
        if 7 in count:
            color7=(0, 255, 0)
        if 8 in count:
            color8=(0, 255, 0)
        if 9 in count:
            color9=(0, 255, 0)
        if 10 in count:
            color10=(0, 255, 0)
        if 11 in count:
            color11=(0, 255, 0)
        if 12 in count:
            color12=(0, 255, 0)
        if 13 in count:
            color13=(0, 255, 0)
        if 14 in count:
            color14=(0, 255, 0)
        if 15 in count:
            color15=(0, 255, 0)
        if 16 in count:
            color16=(0, 255, 0)
        cv2.putText( im, "1", (200, 280), cv2.FONT_HERSHEY_PLAIN, 8, color1, 8 )
        cv2.rectangle(im, (0, 135), (477, 337), color1, 2)
        cv2.putText( im, "2", (677, 280), cv2.FONT_HERSHEY_PLAIN, 8, color2, 8 )
        cv2.rectangle(im, (477, 135), (954, 337), color2, 2)
        cv2.putText( im, "3", (1154, 280), cv2.FONT_HERSHEY_PLAIN, 8, color3, 8 )
        cv2.rectangle(im, (954, 135), (1431, 337), color3, 2)
        cv2.putText( im, "4", (1631, 280), cv2.FONT_HERSHEY_PLAIN, 8, color4, 8 )
        cv2.rectangle(im, (1431, 135), (1908, 337), color4, 2)
        cv2.putText( im, "5", (200, 482), cv2.FONT_HERSHEY_PLAIN, 8, color5, 8 )
        cv2.rectangle(im, (0, 337), (477, 539), color5, 2)
        cv2.putText( im, "6", (677, 482), cv2.FONT_HERSHEY_PLAIN, 8, color6, 8 )
        cv2.rectangle(im, (477, 337), (954, 539), color6, 2)
        cv2.putText( im, "7", (1154, 482), cv2.FONT_HERSHEY_PLAIN, 8, color7, 8 )
        cv2.rectangle(im, (954, 337), (1431, 539), color7, 2)
        cv2.putText( im, "8", (1631, 482), cv2.FONT_HERSHEY_PLAIN, 8, color8, 8 )
        cv2.rectangle(im, (1431, 337), (1908, 539), color8, 2)
        cv2.putText( im, "9", (200, 684), cv2.FONT_HERSHEY_PLAIN, 8, color9, 8 )
        cv2.rectangle(im, (0, 539), (477, 741), color9, 2)
        cv2.putText( im, "10", (630, 684), cv2.FONT_HERSHEY_PLAIN, 8, color10, 8 )
        cv2.rectangle(im, (477, 539), (954, 741), color10, 2)
        cv2.putText( im, "11", (1107, 684), cv2.FONT_HERSHEY_PLAIN, 8, color11, 8 )
        cv2.rectangle(im, (954, 539), (1431, 741), color11, 2)
        cv2.putText( im, "12", (1584, 684), cv2.FONT_HERSHEY_PLAIN, 8, color12, 8 )
        cv2.rectangle(im, (1431, 539), (1908, 741), color12, 2)
        cv2.putText( im, "13", (153, 886), cv2.FONT_HERSHEY_PLAIN, 8, color13, 8 )
        cv2.rectangle(im, (0, 741), (477, 943), color13, 2)
        cv2.putText( im, "14", (630, 886), cv2.FONT_HERSHEY_PLAIN, 8, color14, 8 )
        cv2.rectangle(im, (477, 741), (954, 943), color14, 2)
        cv2.putText( im, "15", (1107, 886), cv2.FONT_HERSHEY_PLAIN, 8, color15, 8 )
        cv2.rectangle(im, (954, 741), (1431, 943), color15, 2)
        cv2.putText( im, "16", (1584, 886), cv2.FONT_HERSHEY_PLAIN, 8, color16, 8 )
        cv2.rectangle(im, (1431, 741), (1908, 943), color16, 2)

def Judging_the_direction(x,z):
    global count
    if 1 not in count and x > 500 and x < 600 and z > 100 and z < 150 : count.append(1)
    elif 2 not in count  and  x > 350 and x < 400 and z > 100 and z < 150 : count.append(2)
    elif 3 not in count  and  x > 200 and x < 300 and z > 100 and z < 150 : count.append(3)
    elif 4 not in count  and  x > 100 and x < 200 and z > 100 and z < 150 : count.append(4)
    elif 5 not in count  and  x > 500 and x < 600 and z > 150 and z < 250 : count.append(5)
    elif 6 not in count  and  x > 350 and x < 400 and z > 150 and z < 250 : count.append(6)
    elif 7 not in count  and  x > 200 and x < 300 and z > 150 and z < 250 : count.append(7)
    elif 8 not in count  and  x > 100 and x < 200 and z > 150 and z < 250 : count.append(8)
    elif 9 not in count  and  x > 500 and x < 600 and z > 250 and z < 300 : count.append(9)
    elif 10 not in count  and   x > 350 and x < 400 and z > 250 and z < 300 : count.append(10)
    elif 11 not in count  and  x > 200 and x < 300 and z > 250 and z < 300 : count.append(11)
    elif 12 not in count  and  x > 100 and x < 200 and z > 250 and z < 300 : count.append(12)
    elif 13 not in count  and  x > 500 and x < 600 and z > 300 and z < 400 : count.append(13)
    elif 14 not in count  and  x > 350 and x < 400 and z > 300 and z < 400 : count.append(14)
    elif 15 not in count  and  x > 200 and x < 300 and z > 300 and z < 400 : count.append(15)
    elif 16 not in count  and  x > 100 and x < 200 and z > 300 and z < 400 : count.append(16)

    return count[int(len(count))-1] 

    
if __name__ == '__main__':
    
    # rtsp://admin:ts123456@10.20.21.240:554
    cap = cv2.VideoCapture(0)
    count_T = 1
    name = input("請輸入學號：")
    while True:
        start_time = time.time()
        
        # Read Image
        ret, im = cap.read()
        image_save = im.copy()
        if ret != True:
            # print('read frame failed')
            continue
        size = im.shape
        
        if size[0] > 700:
            h = size[0] / 3
            w = size[1] / 3
            im = cv2.resize( im, (int( w ), int( h )), interpolation=cv2.INTER_CUBIC )
            size = im.shape
     
        ret, image_points = get_image_points(im)
        if ret != 0:
            # print('get_image_points failed')
            continue
        
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size, image_points)
        if ret != True:
            # print('get_pose_estimation failed')
            continue
        used_time = time.time() - start_time
        # print("used_time:{} sec".format(round(used_time, 3)))
        
        # ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        # euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
        
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
         
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
         
        cv2.circle(im, ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])) , 3, (255,0,0), -1) 
        # Display image
        im = cv2.flip(im, 1)
        im = cv2.resize(im, (1920 ,1080),interpolation=cv2.INTER_CUBIC)
        path_photos_from_camera = './data/data_faces_from_camera/'
        kk = cv2.waitKey(1)

        if name != "":
            count_time = Judging_the_direction(int(nose_end_point2D[0][0][0]),int(nose_end_point2D[0][0][1]))  
            color_change(count_time,"green")
            if count_bef != count_time:

                count_error.remove(count_time)
                count_bef=count_time
                current_face_dir = path_photos_from_camera + "person_" + str(count_T)
                if not os.path.isdir(current_face_dir):
                    os.makedirs(current_face_dir)
                aaa=current_face_dir+"/img_face_1.jpg"
                count_T = count_T +1
                
                cv2.imwrite(aaa, image_save)
            
            if count_time == 0:
                color_change(0,"red")  
            if count_T == 10 : break
        cv2.imshow("Output", im)
        
        image_save
        if kk == ord('q'):
            break
    print("學號：{:}   編號：{:}角度未拍攝成功".format(name,count_error))
    # print('更新完畢') 
    pic_features.main(name)  

            
