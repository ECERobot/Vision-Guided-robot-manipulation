import open3d as o3d
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import random
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
from transforms import *
def loadModel(p: str, task = 'multi-task'):
    model = YOLO(p, task='multi-task')
    return model
def model_inference(model, image, thres = 0.25):
    r =model.predict(source = image,stream=False,conf = thres, save_txt  =False, verbose = False)[0]
    if r.masks is None:
        return None
    mask_flattend  =r.masks.data.cpu().numpy()
    
    keypoints = Results(orig_img=r.orig_img,path=r.path,names= r.names,boxes=r.boxes.data,keypoints=r.keypoints.data).keypoints
    out_img = Results(orig_img=r.orig_img,path=r.path,names= r.names,boxes=r.boxes.data, masks =r.masks.data, keypoints=r.keypoints.data).plot()
    kpts =  keypoints.xy.cpu().numpy()
    # print("Key points:", kpts)
    # masks.int().cpu().numpy().astype('uint8')
    masks = (mask_flattend*255).astype(np.uint8)
    return (out_img, masks, kpts) # output image, masks, keypoints
# def find_h_w(aligned_points):
def euclidean_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
def find_h_w(aligned_points):
    # Points 0 to 1 (top-left to top-right)
    distance_01 = euclidean_distance(aligned_points[0], aligned_points[1])
    distance_03 = euclidean_distance(aligned_points[0], aligned_points[3])
    # Compare lengths
    if distance_01 > distance_03:
        # print("Distance from point 0 to point 1 is greater than from point 0 to point 3.")
        h , w = 1.1, 0.7
    elif distance_01 < distance_03:
        # print("Distance from point 0 to point 3 is greater than from point 0 to point 1.")
        h , w = 0.7 , 1.1
    else:
        # print("Distance from point 0 to point 1 is equal to distance from point 0 to point 3.")
        w,h = 1,1
    return ( h,w)
def pose_estimation(image_color, masks, kpts, K,  intrinsics, depth_image):
    if masks is None:
        return None
    retrun_grasp_pts = []  
    for  i, (mask, kpt) in enumerate(zip(masks, kpts)):

        kernel_size = 11  # You can adjust this size
        # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
	    # closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        # Perform the erosion
        eroded_mask = cv2.erode(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
        else:
            print("No contours found!")
       
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # Convert to integer coordinates
        # cv2.drawKeypoints(mask, box,0, ( 255), 
        #                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # cv2.drawContours(image_color, [box], contourIdx=-1, color=(255,255,0), thickness=2)
        for point in box:
            cv2.circle(image_color, tuple(point), radius=5, color=(255, 255, 0), thickness=-1)  # Red points
        aligned_points = reorder_points(box)
        # draw_keypoint(image_color, aligned_points)
        points = aling_keypoint(kpt, eroded_mask)
        # print("Points after aligment:", points)
        draw_keypoint(image_color, points)
        check_condition = has_zero_points(points)
        if check_condition:
            projected_point = aligned_points
            (height_c, weight_c) = find_h_w(projected_point)
        else:
            projected_point = points
            (height_c, weight_c) = find_h_w(projected_point)
        # print("(height_c, weight_c):", (height_c, weight_c))
        homo_matrix = pose_estimation_PNP(projected_point, K, height_c, weight_c)
        visual_img =  visualiz_bbox_3D(image_color, box, homo_matrix, K, height_c, weight_c)
        # Define the 3D axes in the world (endpoints of the axes)
        axis_3D = np.float32([[1, 0, 0],  # X-axis (red)
                            [0, 1, 0],  # Y-axis (green)
                            [0, 0, 1],  # Z-axis (blue)
                            [0, 0, 0]]) # Origin

        # Project the 3D points to 2D
       # Define the 3D axes (X, Y, Z)
        axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(-1, 3)

        # Project 3D axis points to 2D image points using the camera intrinsic matrix and pose
        img_points, _ = cv2.projectPoints(axis, homo_matrix[1], homo_matrix[2], K, None)
        grasp_2d = np.mean(box, axis=0)
        grasp_3d = compute3D(grasp_2d.astype(int), intrinsics, depth_image)
        Euler_angle = rotation_matrix_to_euler_angles(homo_matrix[0])
        retrun_grasp_pts.append((grasp_3d[0][0], grasp_3d[0][1], grasp_3d[0][2], Euler_angle[0], Euler_angle[1], Euler_angle[2]))
        
        # # Draw the 3D axes on the image
        # origin = tuple(grasp_2d.astype(int))  # Ensure the points are integers
        # x_axis = tuple(img_points[0].ravel().astype(int))  # X axis
        # y_axis = tuple(img_points[1].ravel().astype(int))  # Y axis
        # z_axis = tuple(img_points[2].ravel().astype(int))  # Z axis

        # image = cv2.arrowedLine(visual_img, origin, x_axis, (0, 0, 255), 3)  # X axis in Red
        # image = cv2.arrowedLine(visual_img, origin, y_axis, (0, 255, 0), 3)  # Y axis in Green
        # image = cv2.arrowedLine(visual_img, origin, z_axis, (255, 0, 0), 3)  # Z axis in Blue
        # merg_img = np.concatenate((out_img, visual_img), axis = 1)
    return visual_img, retrun_grasp_pts, homo_matrix
def show(pcPoints, pcColors):
    """
    Visualizes the generated 3D point cloud using Open3D.
    """
    if pcPoints is not None and pcColors is not None:
        points = pcPoints.reshape(-1, 3)
        colors = pcColors.reshape(-1, 3) 
        colors = np.asarray(colors/255) # rescale to 0 to 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
    else:
        print("No 3D points or colors to visualize.")
def write(pathName,points, colors):
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    verts = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3) 
    verts = np.hstack([verts, colors])
    with open(pathName, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
def resizeImage(img, percent):
    """
    Resize an image by a given percentage.

    Args:
        img (numpy.ndarray): Input image to be resized.
        percent (float): Percentage by which to resize the image.

    Returns:
        numpy.ndarray: Resized image.
    """
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized
def read_stereo_paremater(path: str):
    get_parameter = cv2.FileStorage(path, cv2.FileStorage_READ)
    Q = get_parameter.getNode('q').mat()
    R_x = get_parameter.getNode('stereoMapR_x').mat()
    R_y = get_parameter.getNode('stereoMapR_y').mat()
    L_x = get_parameter.getNode('stereoMapL_x').mat()
    L_y = get_parameter.getNode('stereoMapL_y').mat()
    P1 = get_parameter.getNode('p1').mat()
    get_parameter.release()
    return np.array([R_x, R_y, L_x, L_y, Q, P1], dtype=object)
def read_img(path_img: str, color = True):
    color_img  = cv2.imread(path_img)
    gray_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    if color:
        re_img = color_img
    else:
        re_img = gray_color
    return re_img
def remap_img(image, map1, map2):
    rectified_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return rectified_image
def show_img(window_name: str, image: np.array, wait_time: int):
    cv2.imshow(f"{window_name}", image)
    cv2.waitKey(wait_time)
def compute_grasp_params(image_points, depth_frame, intrinsics):
    # Convert 2D points to 3D points
    points_3d = []
    for u, v in image_points:
        depth = depth_frame.get_distance(u, v)
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
        points_3d.append(point_3d)
    
    points_3d = np.array(points_3d)  # Convert to NumPy array
    
    # Calculate grasp point (centroid)
    grasp_point = np.mean(points_3d, axis=0)
    
    # Calculate surface normal (grasp direction)
    vector1 = points_3d[1] - points_3d[0]
    vector2 = points_3d[2] - points_3d[0]
    normal = np.cross(vector1, vector2)
    grasp_direction = -normal / np.linalg.norm(normal)  # Normalize and invert for approach direction
    
    return grasp_point, grasp_direction
# def compute_3D_point(image_point, depth_frame, intrinsics):
#      # Convert 2D points to 3D points
#     points_3d = []
#     u, v = image_point
#     depth = depth_frame.get_distance(u, v)
#     point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
#     points_3d.append(point_3d)
    
#     points_3d = np.array(points_3d)  # Convert to NumPy array
#     return points_3d

def writetxt(file_name,my_array):
    # Open the file in write mode
    with open(file_name, "w") as file:
        # Iterate through the subarrays and write them to the file
        for subarray in my_array:
            arr = []
            for ai, a in enumerate(subarray):
                arr.append(round(a,5))
            # Convert the subarray to a string and write it to the file
            file.write(" ".join(map(str, arr)) + "\n")
def plane_3points(P1, P2, P3):
    """
    Calculate the plane equation parameters (a, b, c, d) from three 3D points.
    """
    a = (P2[1] - P1[1]) * (P3[2] - P1[2]) - (P3[1] - P1[1]) * (P2[2] - P1[2])
    b = (P2[2] - P1[2]) * (P3[0] - P1[0]) - (P3[2] - P1[2]) * (P2[0] - P1[0])
    c = (P2[0] - P1[0]) * (P3[1] - P1[1]) - (P3[0] - P1[0]) * (P2[1] - P1[1])
    d = -(a * P1[0] + b * P1[1] + c * P1[2])
    return np.array([a, b, c, d], dtype=np.float32)

def angle_3points(P1, P2, P3):
    """
    Calculate the angles (Rx, Ry, Rz) based on three 3D points.
    """
    P1, P2, P3 = np.array(P1), np.array(P2), np.array(P3)

    # Compute the centroid of the points
    Pc = (P1 + P2 + P3) / 3

    # Vectors between points
    v1 = P2 - P1
    v2 = P3 - P1

    # Normal vector of the plane
    n = -np.cross(v1, v2)
    n = n / np.linalg.norm(n)

    # Calculate P4
    P4 = np.array([(P1[0] + P2[0]) / 2, Pc[1], 0], dtype=np.float32)
    pl = plane_3points(P1, P2, P3)

    if pl[2] != 0:  # Avoid division by zero
        P4[2] = (-pl[3] - pl[0] * P4[0] - pl[1] * P4[1]) / pl[2]
    else:
        raise ValueError("The plane equation is degenerate (c=0).")

    # Direction vectors
    o = P4 - Pc
    o = o / np.linalg.norm(o)
    bn = np.cross(n, o)

    # Compute angles in radians
    Rx = np.arctan2(o[2], n[2])
    Ry = -np.arctan(bn[2])
    Rz = np.arctan2(o[1], o[0])

    # Convert radians to degrees
    angles = np.array([Rx, Ry, Rz]) * 180 / np.pi
    return angles
def find_corners_from_mask(mask):
    """
    Find the 4 corners of a planar object given its mask.

    Parameters:
        mask (ndarray): Binary mask of the object (0 background, 255 object).

    Returns:
        corners (ndarray): 4x2 array of corner points (x, y).
    """
    # Step 1: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Find the largest contour (assuming the object is the largest blob)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # Step 3: Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)  # 2% of the contour's perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Step 4: Ensure the polygon has 4 corners
    if len(approx) == 4:
        corners = approx.reshape(4, 2)  # Reshape to (4, 2)
        # Step 5: Sort corners in a consistent order (top-left, top-right, bottom-right, bottom-left)
        rect = sort_corners(corners)
        return rect
    else:
        raise ValueError(f"Expected 4 corners, but found {len(approx)} points.")

def sort_corners(corners):
    """
    Sort corners in the order: top-left, top-right, bottom-right, bottom-left.

    Parameters:
        corners (ndarray): 4x2 array of corner points (x, y).

    Returns:
        sorted_corners (ndarray): Sorted 4x2 array of corner points.
    """
    # Sort by y-coordinates, then split into top and bottom pairs
    corners = sorted(corners, key=lambda x: (x[1], x[0]))
    top = sorted(corners[:2], key=lambda x: x[0])  # Sort top points by x
    bottom = sorted(corners[2:], key=lambda x: x[0])  # Sort bottom points by x
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
def pose_estimation_PNP(image_points, K, height, weight):
    image_points =image_points.astype(np.float32)
    # '''x value of Position 0 is greater than x value of position 3''' 
    object_points = np.array([
        [0, 0, 0],
        [height, 0, 0],
        [height, weight, 0],
        [0, weight, 0]
    ], dtype=np.float32)
    '''x value of Position 0 is less than x value of position 3'''
    # object_points = np.array([
    #     [0, weight, 0],
    #     [height, weight, 0],
    #     [height, 0, 0],
    #     [0, 0, 0]
    # ], dtype=np.float32)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, K, None) #cv2.SOLVEPNP_EPNP
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    return (rotation_matrix, rotation_vector, translation_vector) # Return [R/T]
def visualiz_bbox_3D(color_image, image_points, homo_matrix, K, height, weight):
    image_points =image_points.astype(np.float32)
    rotation_matrix,_, translation_vector = homo_matrix
     # Visualize Bounding Box
    bbox_3d = np.array([
        [0, 0, 0], [height, 0, 0], [height, weight, 0], [0, weight, 0],  # Bottom face
        [0, 0, -0.2], [height, 0, -0.2], [height, weight, -0.2], [0, weight, -0.2]  # Top face
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
    return color_image
# Reorder the points
def reorder_points(points):
    points = sorted(points, key=lambda p: (p[1], p[0]))  # Sort by y, then x
    top_two = sorted(points[:2], key=lambda p: p[0])  # Top points sorted by x
    bottom_two = sorted(points[2:], key=lambda p: p[0])  # Bottom points sorted by x
    return np.array([top_two[0], top_two[1], bottom_two[1], bottom_two[0]])
def compute3D(image_point, intr, depth_mat):
    """
    Compute 3D points for specified 2D points on a planar object using RealSense intrinsics.
    :param intr: RealSense intrinsics (rs.intrinsics)
    :param color_mat: Color image (numpy array)
    :param depth_mat: Depth image (numpy array)
    :return: 3D points (5x3 numpy array)
    """
    # Define 2D points
    # pC = (320, 240)  # Center
    # pU = (320, 220)  # Up
    # pD = (320, 260)  # Down
    # pL = (300, 240)  # Left
    # pR = (340, 240)  # Right

    points_2D = [image_point]
    # print("image_point",image_point)

    # Rectangles around each point (for averaging depth values)
    rectangles = [(x - 5, y - 5, 10, 10) for (x, y) in points_2D]

    # Ensure rectangles are within image bounds
    depth_height, depth_width = depth_mat.shape
    rectangles = [
        (max(0, x), max(0, y), min(10, depth_width - x), min(10, depth_height - y))
        for (x, y, w, h) in rectangles
    ]

    # Compute average depth for each rectangle
    depths = []
    for (x, y, w, h) in rectangles:
        roi = depth_mat[y:y+h, x:x+w]
        roi = roi[roi != 0]
        # print("np.mean(roi):", roi)
        depths.append(np.mean(roi))

    # Convert to 3D points using RealSense intrinsics
    points_3D = []
    for i, (x, y) in enumerate(points_2D):
        depth = depths[i]
        point_3D = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth)
        # print("point_3D", point_3D)
        points_3D.append([point_3D[0]-17,point_3D[1], point_3D[2]] )

    # Convert to numpy matrix (5x3)
    planar_point_3D = np.array(points_3D)  # Scale to millimeters
    # # Draw rectangles on the color image
    # for (x, y, _, _), (rect_x, rect_y, rect_w, rect_h) in zip(points_2D, rectangles):
    #     color = (0, 255, 0) if (x, y) == pC else (0, 0, 255)
    #     cv2.rectangle(color_mat, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), color, 1)

    return planar_point_3D
def aling_keypoint(keypoints, mask):
    if mask is None or keypoints is None:
        return None
    keypoints = align_points_to_mask_center(keypoints, mask)
    # Ensure keypoints lie on the mask
    aligned_keypoints = []
    for kp in keypoints:  
        x, y = int(kp[0]), int(kp[1])
        if mask[y, x] > 0:  # Keypoint lies on the mask
            aligned_keypoints.append((x, y))
        else:
            # Snap keypoint to nearest mask point
            distance_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
            nearest_y, nearest_x = np.unravel_index(np.argmin(distance_transform), distance_transform.shape)
            aligned_keypoints.append((nearest_x, nearest_y))
    # Step 1: Find the bounding rectangle of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Assume the largest contour is the mask
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Step 2: Define the corners of the desired square
        square_corners = box
        # print("Box,", box)
        # print("aligned_keypoints", aligned_keypoints)
        aligned_keypoints = np.array(aligned_keypoints, dtype=np.float32)
        square_corners = np.array(square_corners, dtype=np.float32)
        # Step 3: Compute perspective transformation
        transform_matrix = cv2.getPerspectiveTransform(aligned_keypoints, square_corners)

        # Step 4: Transform the points to align with the square
        aligned_points = cv2.perspectiveTransform(aligned_keypoints[None, :, :], transform_matrix)[0]
    return aligned_points
def draw_keypoint(image, keypoints):
    # Draw keypoints on the image
    for i, (x, y) in enumerate(keypoints):
        # Convert to integer for drawing
        x, y = int(x), int(y)
        
        # Generate a random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Draw a circle at the keypoint
        cv2.circle(image, (x, y), radius=5, color=color, thickness=-1)  # Circle with random color
    
        # Optional: Add a label with the index
        cv2.putText(image, f'{i}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return image
def has_zero_points(points):
    for point in points:
        if point[0] == 0 and point[1] == 0:  # Check if both coordinates are zero
            return True
    return False
def align_points_to_mask_center(keypoints, mask):
    if mask is None or keypoints is None or len(keypoints) != 4:
        return None
    
    # Ensure keypoints are in np.float32 format
    keypoints = np.array(keypoints, dtype=np.float32)

    # Step 1: Calculate the center of the 4 points
    center_points = np.mean(keypoints, axis=0)

    # Step 2: Calculate the center of the mask using moments
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return None  # No valid area in the mask
    cx_mask = int(moments["m10"] / moments["m00"])
    cy_mask = int(moments["m01"] / moments["m00"])

    # Step 3: Compute translation vector
    translation_vector = np.array([cx_mask, cy_mask]) - center_points

    # Step 4: Apply translation to all 4 points
    aligned_keypoints = keypoints + translation_vector

    return aligned_keypoints
def rotation_matrix_to_euler_angles(R):
    # Check if the rotation matrix is valid
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")

    # Calculate pitch, yaw, and roll (assuming XYZ convention)
    pitch = np.arctan2(R[2, 1], R[2, 2])  # Rotation around Y-axis (pitch)
    yaw = np.arcsin(-R[2, 0])             # Rotation around Z-axis (yaw)
    roll = np.arctan2(R[1, 0], R[0, 0])   # Rotation around X-axis (roll)

    return (np.degrees(pitch), np.degrees(yaw), np.degrees(roll))
def convert_camera_robot(posture, robotPose):
    cam2end = np.array([[-0.999856, 0.0168687,0.00182544,17.9595],
                        [-0.0168131,-0.999493,0.0270459,62.2374],
                        [0.00228074,0.0270113,0.999633,53.5254],
                        [0,0,0,1]]) 
    end2tool = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,125],
                 [0,0,0,1]])
    end2tool_ = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1, -125],
                 [0,0,0,1]])  
    ob2cam = euler2Rotation(posture)
    robotPose_ro = euler2Rotation(robotPose)
    # print("robotPose", robotPose)
    # print("np.linalg.inv(end2tool)", np.linalg.inv(end2tool))
    # print("ob2cam", ob2cam)
    A = robotPose_ro @ np.linalg.inv(end2tool) @ cam2end
    # A = robotPose_ro@ cam2end
    # print("A", A)
    B = A @ ob2cam
    # print("B", B)
    # target = B @ end2tool_
    target = B 
    return target
# Define a function to map pixels to 3D space
def deproject_to_point(x, y, depth, intrinsics):
    return rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
def segment_mask_point_cloud(color_image, instance_masks, depth_intrinsics, depth_image):
    # Create a numpy array for storing 3D points and color values
    point_clouds = []
    point_cloud_color = []
    # Assume instance_mask is a binary mask with the same dimensions as the images
    for  i, mask in enumerate(instance_masks):
        kernel_size = 5 # You can adjust this size
        # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
	    # closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        # Perform the erosion
        eroded_mask = cv2.erode(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area < 20000:
                continue
        else:
            print("No contours found!")
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # Convert to integer coordinates
        # cv2.drawKeypoints(mask, box,0, ( 255), 
        #                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # cv2.drawContours(color_image, [box], contourIdx=-1, color=(255,255,0), thickness=2)
        instance_mask = np.ones((480, 640), dtype=np.uint8)
        cv2.drawContours(instance_mask, [largest_contour], contourIdx=-1, color=(255,255,0), thickness=cv2.FILLED)
        # cv2.imshow("mask", instance_mask)
        # cv2.waitKey(0)
        masked_depth = np.where(instance_mask == 255, depth_image, 0)
        # Loop over each pixel in the depth image and convert it to 3D
        height, width = masked_depth.shape
        points = []
        colors = []
        for v in range(height):
            for u in range(width):
                # Get the depth value at pixel (u, v)
                depth = masked_depth[v, u]
                if depth == 0:  # Skip invalid depth
                    continue
                # Convert (u, v, depth) to 3D point in camera coordinates
                point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                # Append the 3D point and the corresponding color
                points.append(point)
                colors.append(color_image[v, u] / 255.0)  # Normalize color to [0, 1] for Open3D
        # Convert points and colors to numpy arrays
        points = np.array(points)
        colors = np.array(colors)
        point_clouds.append(points)
        point_cloud_color.append(colors)
    return point_clouds, point_cloud_color
def get_point_cloud(color_image,depth_intrinsics, depth_image):
    height, width = depth_image.shape
    points = []
    colors = []
    for v in range(height):
        for u in range(width):
            # Get the depth value at pixel (u, v)
            depth = depth_image[v, u]
            if depth == 0:  # Skip invalid depth
                continue
            # Convert (u, v, depth) to 3D point in camera coordinates
            point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
            # Append the 3D point and the corresponding color
            points.append(point)
            colors.append(color_image[v, u] / 255.0)  # Normalize color to [0, 1] for Open3D
    # Convert points and colors to numpy arrays
    points = np.array(points)
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def show_pc_seg2(points,colors):
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    # Set the points and colors in the Open3D point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])
def show_pc_seg(point_clouds, pc_color):
    if point_clouds is not None:
         # Extract point cloud vertices
        verts = np.asanyarray(point_clouds).view(np.float32).reshape(-1, 3)  # xyz coordinates
        colors = np.array(pc_color, dtype=np.float64).reshape(-1,3)
        print("Verts shape:", verts.shape)
        print("Colors shape:", colors.shape)
        assert verts.shape[0] == colors.shape[0], "Mismatch in number of points and colors!"
        # Create a point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
    else:
        print("None point cloud")
def show_pc_realsene(color_frame, depth_frame):
    # Convert depth frame to a numpy array
    # depth_image = np.asanyarray(depth_frame.get_data())

     # Map depth frame to point cloud
    pc = rs.pointcloud()
    pc.map_to(color_frame)  # Map the color frame to the depth frame
    points = pc.calculate(depth_frame)

    # Extract point cloud vertices
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz coordinates

    # Extract texture coordinates
    tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    color_image = np.asanyarray(color_frame.get_data())
    # Map texture coordinates to RGB colors
    h, w, _ = color_image.shape
    colors = []
    for uv in tex_coords:
        u = int(uv[0] * w)
        v = int(uv[1] * h)
        if 0 <= u < w and 0 <= v < h:
            colors.append(color_image[v, u] / 255.0)  # Normalize RGB to [0, 1]
        else:
            colors.append([0, 0, 0])  # Black for out-of-bound points
    colors = np.array(colors, dtype=np.float64).reshape(-1,3)
    
    # Ensure points and colors are compatible
    assert verts.shape[0] == colors.shape[0], "Mismatch in number of points and colors!"
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    return pcd

def draw_bounding_box_on_pcd(grasp3d, point_clouds, colors, homo_matrix):
    """
    Draws a bounding box on the point cloud based on a homogeneous transformation matrix.
    
    Args:
        point_clouds: The 3D points of the point cloud.
        colors: The colors associated with each point in the point cloud.
        homo_matrix: 4x4 homogeneous transformation matrix containing rotation and translation.
    """
    # Apply the homogeneous transformation (rotation + translation) to the bounding box corners
    # rotation_matrix =homo_matrix[0]  # Extract the 3x3 rotation matrix
    # translation_vector = homo_matrix[1]  # Extract the translation vector
    # Create point cloud object for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #  Remove noise:
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)
    # Center point
     # OBB 
    obb = inlier_cloud.get_oriented_bounding_box()
    obb.color = (1, 0, 0)  # Color the bounding box red for visibility
    center = obb.center
    rotation_matrix = obb.R 
    angle_el = rotation_matrix_to_euler_angles(rotation_matrix)
    print("angle_el", angle_el)
    # center = [grasp3d[0][0], grasp3d[0][1], grasp3d[0][2]]
    axes_points = np.array([
        center,  # Origin (center of the bounding box)
        center + rotation_matrix @ np.array([100, 0, 0]),  # X-axis
        center + rotation_matrix @ np.array([0, 100, 0]),  # Y-axis
        center + rotation_matrix @ np.array([0, 0, -100])   # Z-axis
    ])
    # Create lines for the coordinate axes (X, Y, Z)
    lines = [
        [0, 1],  # X-axis line
        [0, 2],  # Y-axis line
        [0, 3]   # Z-axis line
    ]
    # Create Open3D LineSet to visualize the coordinate axes
    coordinate_axes = o3d.geometry.LineSet()
    coordinate_axes.points = o3d.utility.Vector3dVector(axes_points)
    coordinate_axes.lines = o3d.utility.Vector2iVector(lines)
    # Define colors for the lines
    colors = [
        [1, 0, 0],  # Red for X-axis
        [0, 1, 0],  # Green for Y-axis
        [0, 0, 1]   # Blue for Z-axis
    ]
    coordinate_axes.colors = o3d.utility.Vector3dVector(colors)
   
    # obb.rotate(rotation_center.T, center = center_rotation)
    # obb.rotate(rotation_matrix, center = center_rotation)
    mesh_r = o3d.geometry.TriangleMesh.create_coordinate_frame(0.04,origin = obb.get_center())
    mesh_r.rotate(obb.R, obb.get_center())
    # o3d.visualization.draw_geometries([pcd, obb, mesh_r])

    # Visualize the point cloud and the coordinate axes
    # o3d.visualization.draw_geometries([coordinate_axes, inlier_cloud, obb, mesh_r])
    return coordinate_axes, obb
def pc_plane_seg_ransac(point_clouds,colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_size = 0.005
    pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    plane_model, inliers = pcd_sampled.segment_plane(distance_threshold=0.1,
                                         ransac_n=3,
                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    obb = inlier_cloud.get_oriented_bounding_box()
    obb.color = (1, 0, 0)  # Color the bounding box red for visibility
    center = obb.center
    rotation_matrix = obb.R 
    angle_el = rotation_matrix_to_euler_angles(rotation_matrix)
    print("angle_el", angle_el)
    # center = [grasp3d[0][0], grasp3d[0][1], grasp3d[0][2]]
    axes_points = np.array([
        center,  # Origin (center of the bounding box)
        center + rotation_matrix @ np.array([100, 0, 0]),  # X-axis
        center + rotation_matrix @ np.array([0, 100, 0]),  # Y-axis
        center + rotation_matrix @ np.array([0, 0, -100])   # Z-axis
    ])
    # Create lines for the coordinate axes (X, Y, Z)
    lines = [
        [0, 1],  # X-axis line
        [0, 2],  # Y-axis line
        [0, 3]   # Z-axis line
    ]
    # Create Open3D LineSet to visualize the coordinate axes
    coordinate_axes = o3d.geometry.LineSet()
    coordinate_axes.points = o3d.utility.Vector3dVector(axes_points)
    coordinate_axes.lines = o3d.utility.Vector2iVector(lines)
    # Define colors for the lines
    colors = [
        [1, 0, 0],  # Red for X-axis
        [0, 1, 0],  # Green for Y-axis
        [0, 0, 1]   # Blue for Z-axis
    ]
    coordinate_axes.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, obb])
    return coordinate_axes, obb
    