# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics import YOLO
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
import time
from ultralytics.engine.results import Results
import math
# import supervision as sv
# Input folder
data_name = '/DATASETS/Orchid_22_11_v3'
OUT_P = '//Results'
current_dir = os.getcwd()
path_data = "".join([current_dir,data_name])
path_out = "".join([current_dir,OUT_P])
#-----
save_dir = R'C:\Users\Thinh\Desktop\Robotics\OUTPUT'

if __name__ == "__main__":
  
    threshold = 0.5
    p = R"C:\Users\Thinh\Desktop\ultralytics_multi\runs\multi-task\train\weights\best.pt"
    model = YOLO(p, task='multi-task')
    path_img= sorted(glob.glob(os.path.join(R'C:\Users\Thinh\Desktop\Robotics\captured_images', '*.png')))
    # print(path_img)
    for image in path_img:
        base_name = image.split("\\")[-1]
        # print(base_name)
        r = model.predict(source = f'{image}',stream=False,conf = threshold, save_txt  =False, verbose = False)[0]
        color_b = (255, 0, 0)
        color_r = (0, 0, 255)
        thickness = 2
        img = cv2.imread(image)
        # print("results:", r.mask.data)
        if r.boxes.data is None:
            print("No data")
            continue
        seg_key = Results(orig_img=r.orig_img,path=r.path,names= r.names,boxes=r.boxes.data,keypoints=r.keypoints.data, masks=r.masks.data).plot()
        out_image = os.path.join(save_dir,'{}'.format(base_name))
        cv2.imwrite(out_image,seg_key)
        # cv2.imshow('image', seg_key)
        # k = cv2.waitKey(0)
        # if k == ord('q'):
        #     break
