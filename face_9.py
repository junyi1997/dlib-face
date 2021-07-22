#!/usr/bin/env python
count=[0]  
count_bef=0
count_error=[1,2,3,4,5,6,7,8,9]
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
predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')
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

    
    if color_i == "red":
        cv2.putText( im, "1", (280, 320), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (0, 135), (636, 404), (0, 0, 255), 2)
        cv2.putText( im, "2", (916, 320), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (636, 135), (1272, 404), (0, 0, 255), 2)
        cv2.putText( im, "3", (1552, 320), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (1272, 135), (1908, 404), (0, 0, 255), 2)
        cv2.putText( im, "4", (280, 589), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (0, 404), (636, 674), (0, 0, 255), 2)
        cv2.putText( im, "5", (916, 589), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (636, 404), (1272, 674), (0, 0, 255), 2)
        cv2.putText( im, "6", (1552, 589), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (1272, 404), (1908, 674), (0, 0, 255), 2)
        cv2.putText( im, "7", (280, 858), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (0, 674), (636, 943), (0, 0, 255), 2)
        cv2.putText( im, "8", (916, 858), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (636, 674), (1272, 943), (0, 0, 255), 2)
        cv2.putText( im, "9", (1552, 858), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8 )
        cv2.rectangle(im, (1272, 674), (1908, 943), (0, 0, 255), 2)

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

        cv2.putText( im, "1", (280, 320), cv2.FONT_HERSHEY_PLAIN, 8, color1, 8 )
        cv2.rectangle(im, (0, 135), (636, 404), color1, 5)
        cv2.putText( im, "2", (916, 320), cv2.FONT_HERSHEY_PLAIN, 8, color2, 8 )
        cv2.rectangle(im, (636, 135), (1272, 404), color2, 5)
        cv2.putText( im, "3", (1552, 320), cv2.FONT_HERSHEY_PLAIN, 8, color3, 8 )
        cv2.rectangle(im, (1272, 135), (1908, 404), color3, 5)
        cv2.putText( im, "4", (280, 589), cv2.FONT_HERSHEY_PLAIN, 8, color4, 8 )
        cv2.rectangle(im, (0, 404), (636, 674), color4, 5)
        cv2.putText( im, "5", (916, 589), cv2.FONT_HERSHEY_PLAIN, 8, color5, 8 )
        cv2.rectangle(im, (636, 404), (1272, 674), color5, 5)
        cv2.putText( im, "6", (1552, 589), cv2.FONT_HERSHEY_PLAIN, 8, color6, 8 )
        cv2.rectangle(im, (1272, 404), (1908, 674), color6, 5)
        cv2.putText( im, "7", (280, 858), cv2.FONT_HERSHEY_PLAIN, 8, color7, 8 )
        cv2.rectangle(im, (0, 674), (636, 943), color7, 5)
        cv2.putText( im, "8", (916, 858), cv2.FONT_HERSHEY_PLAIN, 8, color8, 8 )
        cv2.rectangle(im, (636, 674), (1272, 943), color8, 5)
        cv2.putText( im, "9", (1552, 858), cv2.FONT_HERSHEY_PLAIN, 8, color9, 8 )
        cv2.rectangle(im, (1272, 674), (1908, 943), color9, 5)


def Judging_the_direction(x,z):
    # print("X = {:}  Z = {:}".format(x,z))
    # return 0
    global count
    if 1 not in count and x > 450 and x < 600 and z > 100 and z < 160 : count.append(1)
    elif 2 not in count  and  x > 220 and x < 410 and z > 100 and z < 160 : count.append(2)
    elif 3 not in count  and  x > 100 and x < 210 and z > 100 and z < 160 : count.append(3)
    elif 4 not in count  and  x > 450 and x < 600 and z > 180 and z < 270 : count.append(4)
    elif 5 not in count  and  x > 220 and x < 410 and z > 180 and z < 270 : count.append(5)
    elif 6 not in count  and  x > 100 and x < 210 and z > 180 and z < 270 : count.append(6)
    elif 7 not in count  and  x > 450 and x < 600 and z > 320 and z < 400 : count.append(7)
    elif 8 not in count  and  x > 220 and x < 410 and z > 320 and z < 400 : count.append(8)
    elif 9 not in count  and  x > 100 and x < 210 and z > 320 and z < 400 : count.append(9)


    return count[int(len(count))-1] 

    
if __name__ == '__main__':
    
    # rtsp://admin:ts123456@10.20.21.240:554
    cap = cv2.VideoCapture(0)
    count_T = 1
    name = input("請輸入學號：")
    # name = "M10907324"
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
                current_face_dir1 = path_photos_from_camera + "person_10"
                if not os.path.isdir(current_face_dir):
                    os.makedirs(current_face_dir)
                if not os.path.isdir(current_face_dir1):
                    os.makedirs(current_face_dir1)    
                aaa=current_face_dir+"/img_face_1.jpg"
                aaa1=current_face_dir1+"/img_face_{:}.jpg".format(count_T)
                count_T = count_T +1
                
                cv2.imwrite(aaa, image_save)
                cv2.imwrite(aaa1, image_save)

            if count_time == 0:
                color_change(0,"red")  
            if count_T == 10 : break
        cv2.imshow("Output", im)
        
        image_save
        if kk == ord('q'):
            break
    if count_error == []:
        print("學號：{:}   拍攝成功".format(name))
    else:            
        print("學號：{:}   編號：{:}角度未拍攝成功".format(name,count_error))
    # print('更新完畢') 
    pic_features.main(name)  

            
