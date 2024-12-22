import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
import time
from ultralytics.engine.results import Results
import math
from cvfun import *
threshold = 0.5
p = R"C:\Users\Thinh\Desktop\ultralytics_multi\runs\multi-task\train\weights\best.pt"
model = YOLO(p, task='multi-task')
if __name__ == "__main__":
    K = np.array([[615.75, 0, 329.27],
                [0, 616.02, 244.46],
                [0, 0, 1]])
    # # Example: 4 known 2D-3D correspondences (manually labeled for demo)
    image = R'C:\Users\Thinh\Desktop\Robotics\Data\captured_images\image_84.png'
    color_image = cv2.imread(image)
    # r = model.predict(source = f'{image}',stream=False,conf = threshold, save_txt  =False, verbose = False)[0]
    # print("Key points", r.keypoints.data)
    object_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    # third_point_3d = compute_3D_point(third_point.astype(int), depth_frame, intrinsics)
    image_points = np.array([
        [388, 318],  # Example 2D points [388, 318]
        [259, 314],
        [269, 161],
        [394, 170]
    ], dtype=np.float32)

    # image_points = np.array([
    #     [402, 289],  # Example 2D points
    #     [286, 332],
    #     [234, 199],
    #     [352, 151]
    # ], dtype=np.float32)
    # SolvePnP to get rotation and translation
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, K, None, flags= cv2.SOLVEPNP_IPPE_SQUARE
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    # Visualize Bounding Box
    bbox_3d = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, -0.2], [1, 0, -0.2], [1, 1, -0.2], [0, 1, -0.2]  # Top face
    ], dtype=np.float32)
    bbox_3d_transformed = (rotation_matrix @ bbox_3d.T).T + translation_vector.T
    bbox_2d = (K @ bbox_3d_transformed.T).T
    bbox_2d /= bbox_2d[:, 2][:, np.newaxis]

    # Draw bounding box
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        pt1 = tuple(bbox_2d[i, :2].astype(int))
        pt2 = tuple(bbox_2d[j, :2].astype(int))
        cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)
    grasp_2d = np.mean(image_points, axis=0)
    grasp_dir = np.array([0.5, 0.5, -1])
    grasp_dir_3d_transformed = (rotation_matrix @ grasp_dir.T).T + translation_vector.T
    grasp_dir_2d = (K @ grasp_dir_3d_transformed.T).T
    grasp_dir_2d /= grasp_dir_2d[:,2]
    # Draw grasping direction
    cv2.arrowedLine(
        color_image,
        tuple(grasp_2d[:2].astype(int)),
        tuple(grasp_dir_2d[0][:2].astype(int)),
        (255, 0, 0), 2
    )
    cv2.imshow("Bounding Box with Grasping Direction", color_image)
    cv2.waitKey(0)
