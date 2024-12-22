import numpy as np
from Denso_robot.DensoRobotControl import DensoRobotControl
import numpy as np
import pyrealsense2 as rs
import cv2
import time
from ultralytics import YOLO
import math
from cvfun import *  # Assuming this contains helper functions like pose_estimation and model_inference
from transforms import *
# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Camera Intrinsics
K = np.array([[615.75, 0, 329.27],
              [0, 616.02, 244.46],
              [0, 0, 1]])

# Load YOLO model for object detection
path_model = R"C:\Users\Thinh\Desktop\ultralytics_multi\runs\multi-task\train\weights\last.pt"
model_yolo = YOLO(path_model)

# Initialize RealSense alignment
align = rs.align(rs.stream.color)

# Function to capture and process one frame at a time
def capture_and_process_frame():
    # Capture a single set of frames
    frameset = pipeline.wait_for_frames()
    frameset = align.process(frameset)

    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    # # Adjust camera settings
    # device = pipeline.get_active_profile().get_device()
    # sensor = device.query_sensors()[1]  # Assuming color sensor is the second
    # sensor.set_option(rs.option.enable_auto_exposure, 1)  # 1 to enable, 0 to disable
    # # Enable auto exposure
    # sensor.set_option(rs.option.exposure, 500)  # Increase exposure
    # sensor.set_option(rs.option.gain, 16)  # Adjust gain
    # Get camera intrinsics
    # Get the intrinsics of the depth sensor
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    # Ensure frames are valid
    if not color_frame or not depth_frame:
        print("Skipping frame due to missing data.")
        return None, None
    
    # Convert RealSense frame to OpenCV format
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    # time.sleep(5)

    return color_frame, depth_frame, depth_intrinsics

try:
    robot = DensoRobotControl(port="COM4", baud_rate=19200)
    
    while True:
       
        Home_pose = [367, 20, 466, 180, 0 ,90]
        all_target = []
        if robot.isConnected():
            robot.setTimeout(msg_time_ms=100, pos_time_ms=1500000)
            input("enter")
            robot.moveLine(pose=[367, 20, 466, 180, 0 ,90], speed=15, tool=2,verbose=False)
            time.sleep(1)
            # # Capture a single image from RealSense camera
            color_frame, depth_frame, depth_intrinsics = capture_and_process_frame()
            if color_frame is None or depth_frame is None:
                continue  # Skip this loop iteration if frame capture failed
        
            # Convert depth frame to a numpy array
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            # Run object detection with YOLO
            model_out = model_inference(model_yolo, color_image, 0.25)  # Assuming model_inference is defined elsewhere
            
            if model_out is None:
                print("No prediction")
                continue
            # Visualize detections (You can add confidence filtering here if needed)
            visual_img, grasp_3d, homo = pose_estimation(color_image, model_out[1], model_out[2], K, depth_intrinsics, depth_image)  # Assuming pose_estimation is defined elsewhere
            cv2.imshow("Pose Estimation",visual_img)
            key = cv2.waitKey(1)
            # # ------------- Visualize 3D point cloud -------------- 
            # pcloud, pc_colors = segment_mask_point_cloud(color_image,model_out[1], depth_intrinsics, depth_image)
            # # # print("Points", pcloud)
            # # print("grasp_3d", grasp_3d)
            # # coordinate_axes, obb = draw_bounding_box_on_pcd(grasp_3d, pcloud[0], pc_colors[0] , homo)
            # pcd_l = get_point_cloud(color_image, depth_intrinsics, depth_image)
            # coordinate_axes, obb = pc_plane_seg_ransac(pcloud[0], pc_colors[0])
            # o3d.visualization.draw_geometries([ coordinate_axes, obb, pcd_l])
            # ----------------------------------
            for i, posture in enumerate(grasp_3d):    
                # robotPose = robot.getPosition(verbose = False)
                print("Pose of CAMERA:", posture)
                # posture = [-35, -54, 389.45001220703125, 2.2828297681760628, -0.49919653391717494, 2.5408979607531963]
                target = convert_camera_robot(posture, Home_pose)
                target_euler = rotation2Euler(target)
                # all_target.append(target)
                print("Pose of robot and target "f"{i}", target_euler)
                input("enter")
                # robot.moveLine(pose=[350, 20, 570, -180,0,-90], speed=15, tool=0,verbose=False)
                robot.moveLine(pose= target_euler, speed=15, tool=2,verbose=False)
                input("enter")
                robot.moveLine(pose=[367, 20, 466, 180, 0 ,90], speed=15, tool=2,verbose=False)
            time.sleep(1)
            # Show the result on the frame
            cv2.imshow("Pose Estimation",visual_img)
            # Wait for user input to capture the next frame
            key = cv2.waitKey(1)
            if key == cv2.waitKey(1) & 0xFF:  # 'Esc' key to break the loop
                break

            # Optional: Add a small delay between captures (to control frame rate)
            time.sleep(0.1)

finally:
    # Stop the RealSense pipeline when done
    pipeline.stop()
    cv2.destroyAllWindows()
#  obb = inlier_cloud.get_oriented_bounding_box(robust = True)
#     rotation_matrix_obb = obb.R
#     eluer = rotation_matrix_to_euler_angles(rotation_matrix_obb)
#     print("rotation_matrix", eluer)
#     # # obb.rotate(rotation_matrix.T, center=np.array([0, 0, 0]))
#     obb.color = (0,1,0)