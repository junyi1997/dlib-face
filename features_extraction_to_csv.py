# Copyright (C) 2020 coneypo
# SPDX-License-Identifier: MIT

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera
# Mail:     coneypo@foxmail.com

# 从人脸图像文件中提取人脸特征存入 "features_all.csv" / Extract features from images and save into "features_all.csv"

import os
import dlib
from skimage import io
import csv
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
dict_update = dict()
i=1
# 要读取人脸图像文件的路径 / Path of cropped faces
path_images_from_camera = "./data/data_faces_from_camera/"

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("./data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# 返回单张图像的 128D 特征 / Return 128D features for single image
# Input:    path_img           <class 'str'>
# Output:   face_descriptor    <class 'dlib.vector'>
def return_128d_features(path_img):
    img_rd = io.imread(path_img)
    faces = detector(img_rd, 1)

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了, 所以要确保是 检测到人脸的人脸图像拿去算特征
    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        print("no face")
    return face_descriptor


# 返回 personX 的 128D 特征均值 / Return the mean value of 128D face descriptor for person X
# Input:    path_faces_personX       <class 'str'>
# Output:   features_mean_personX    <class 'numpy.ndarray'>
def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for i in range(len(photos_list)):
            # 调用 return_128d_features() 得到 128D 特征 / Get 128D features for single image of personX
            features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
            # 遇到没有检测出人脸的图片跳过 / Jump if no face detected from image
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        print(" >> 文件夾內沒有照片 / Warning: No images in " + path_faces_personX + '/', '\n')

    # 计算 128D 特征的均值 / Compute the mean
    # personX 的 N 张图像 x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=int, order='C')
    return features_mean_personX

def main(name):
    global i
    # 获取已录入的最后一个人脸序号 / Get the order of latest person
    person_list = os.listdir("data/data_faces_from_camera/")
    person_num_list = []
    for person in person_list:
        person_num_list.append(int(person.split('_')[-1]))
    person_cnt = max(person_num_list)

    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in range(person_cnt):
            # Get the mean/average features of face/personX, it will be a list with a length of 128D
            # print(path_images_from_camera + "person_" + str(person + 1))
            features_mean_personX = return_features_mean_personX(path_images_from_camera + "person_" + str(person + 1))
            dict_update[str(i)]=str(features_mean_personX)
            i=i+1
            writer.writerow(features_mean_personX)
    # print("學號：{:}".format(name)) 
    # # print(dict_update)       
    # for i in range(9):
    #     print("第{:}位特徵：{:}\n".format(i+1,dict_update[i]))
    # 引用私密金鑰
    # path/to/serviceAccount.json 請用自己存放的路徑
    cred = credentials.Certificate('./data/data_dlib/serviceAccount.json')

    # 初始化firebase，注意不能重複初始化
    firebase_admin.initialize_app(cred)

    # 初始化firestore
    db = firestore.client()
    doc_ref_visit = db.collection("feature").document(name)
    # doc_ref提供一個set的方法，input必須是dictionary
    doc_ref_visit.set(dict_update)

if __name__ == '__main__':
    name = "M10907324"
    main(name)
       