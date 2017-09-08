import os
import sys

import tensorflow as tf
import numpy as np
import cv2
import dlib

#设置resize图片的大小
size = [32,32]

src_path = os.path.join(os.path.abspath('.'), 'original_data')
dst_path = os.path.join(os.path.abspath('.'), 'face_data')

classes = ['ill_face', 'healthy_face']
writer = tf.python_io.TFRecordWriter('face_test.tfrecords')

detector = dlib.get_frontal_face_detector()

#将ill_face和healthy_face目录下的图片截取人脸部位，转化成32*32大小的图片，再生成tfrecords格式的文件
for index, class_name in enumerate(classes):
    class_path = os.path.join(src_path, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        
        #截取人脸部位并保存
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dets = detector(gray_img,1)

        face = None
        face_resize = None
        
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = img[x1:y1, x2:y2]

        #判断是否检测到人脸部位
        if np.any(face) == None:
            face_resize = cv2.resize(img,(size[0],size[1]))
        else:
            face_resize = cv2.resize(face,(size[0],size[1]))

        #无论是否检测到人脸，都保存检测的结果到32*32大小的图片
        cv2.imwrite(os.path.join(dst_path,class_name,img_name),face_resize)

        img_raw = face_resize.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))

        writer.write(example.SerializeToString())
            
writer.close()



'''
#将healthy_face目录下的图片提取人脸，并resize成32*32大小
for img_name in os.listdir(os.path.join(src_path,'healthy_face')):
    img_path = os.path.join(src_path,'healthy_face',img_name)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dets = detector(gray_img,1)
    face = None
    face_resize = None
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2]
    print(img_name)
    print(np.any(face))
    if np.any(face) == None:
        face_resize = cv2.resize(img,(size[0],size[1]))
    else:
        face_resize = cv2.resize(face,(size[0],size[1]))
    cv2.imwrite(os.path.join(dst_path,'healthy_face',img_name),face_resize)

#将ill_face目录下的图片resize成32*32大小
for img_name in os.listdir(os.path.join(src_path,'ill_face')):
    img_path = os.path.join(src_path,'ill_face',img_name)
    img = cv2.imread(img_path)
    face_resize = cv2.resize(img,(size[0],size[1]))
    cv2.imwrite(os.path.join(dst_path,'ill_face',img_name),face_resize)
'''