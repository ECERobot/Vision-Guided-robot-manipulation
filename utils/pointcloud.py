import os
import numpy as np
import cv2
import open3d as o3d
import time
from concurrent.futures import ThreadPoolExecutor
from utils.cvfunc import resizeImage, cropBBox, cvShow


def mapImage(img, sm, idx):
    """
    Apply stereo map correction to an input image.

    Parameters:
    - img (np.ndarray): Input image to be corrected.
    - sm (list of np.ndarray): List of stereo maps for correction.
    - idx (int): Index indicating which stereo map to use.

    Returns:
    - Corrected image (np.ndarray) after applying the stereo map.
    """
    if idx < 4:
        correctedImg = cv2.remap(img, sm[0], sm[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    else:
        correctedImg = cv2.remap(img, sm[2], sm[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #INTER_LANCZOS4, which is a higher-quality method suitable for situations where quality is critical, such as in stereo vision applications.
    #INTER_LINEAR, which is faster but may produce slightly lower quality results compared to Lanczos.
    return correctedImg

def mapImages(imgs, stereoMaps):
    """
    Apply mapping functions to a list of images using corresponding stereo maps.

    Parameters:
    - imgs (list of np.ndarray): List of input images to be corrected.
    - stereoMaps (list of np.ndarray): List of stereo maps used for correction.

    Returns:
    - List of corrected images (list of np.ndarray) after applying the stereo maps.
    """
    correctedImgs = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, img in enumerate(imgs):
            sm = stereoMaps[i] if i < 4 else stereoMaps[i-4]
            futures.append(executor.submit(mapImage, img, sm, i))
        
        for future in futures:
            correctedImgs.append(future.result())
    return correctedImgs



class PointCloud:
    def __init__(self, imgL, imgR, Q, minDisp, numDisp, imgPoints = None, imgMask = None):
        """
        Initializes the PointCloud object with stereo images, disparity parameters, and optional masks.

        Parameters:
        imgL (np.ndarray): Left stereo image.
        imgR (np.ndarray): Right stereo image.
        Q (np.ndarray): Disparity-to-depth mapping matrix.
        minDisp (int): Minimum disparity value.
        numDisp (int): Number of disparity values.
        imgPoints (list of tuples, optional): Image points for operations like cropping or masking.
        imgMask (np.ndarray, optional): Mask image for region of interest (ROI) operations.

        Attributes:
        w (int): Width of the images.
        h (int): Height of the images.
        pcPoints (np.ndarray): Placeholder for point cloud points.
        pcColors (np.ndarray): Placeholder for point cloud colors.
        """
        self.imgL = imgL
        self.imgR = imgR
        self.Q = Q
        self.minDisp = minDisp
        self.numDisp = numDisp
        self.imgPoints = imgPoints
        self.imgMask = imgMask
        self.w = self.imgL.shape[1]
        self.h = self.imgL.shape[0]
        self.pcPoints = None
        self.pcColors = None

    def checkerboardCorners(self, squaresSize, boardSize, show):
        """
        Detects chessboard corners in the stereo images.

        Parameters:
        squaresSize (int): Size of the chessboard squares in mm.
        boardSize (tuple): Number of internal corners in (columns, rows).
        show (bool): Flag to display the detected corners.

        Returns:
        tuple: Detected corners in the left and right images (cornersL, cornersR).
        """
        imgL = self.imgL 
        imgR = self.imgR 
        grayL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
        objPoints = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
        objPoints[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
        objPoints *= squaresSize  # size_of_chessboard_squares_mm
        retL, cornersL = cv2.findChessboardCorners(grayL, boardSize, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        retR, cornersR = cv2.findChessboardCorners(grayR, boardSize, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if retL and retR:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(imgL, boardSize, cornersL, retL)
            cv2.drawChessboardCorners(imgR, boardSize, cornersR, retR) 
            mergeImg = np.concatenate((imgL, imgR), axis=1)
            if show:
                cv2.imshow("Checkerboard corners", resizeImage(mergeImg,20))
                cv2.waitKey(0)
            return cornersL, cornersR
        else:
            print("Cannot detect")
            return None, None
        
    @staticmethod    
    def disparity(pointsL, pointsR):
        """
        Computes disparity between corresponding points in the left and right images.

        Parameters:
        pointsL (list of tuples): Detected corners in the left image.
        pointsR (list of tuples): Detected corners in the right image.

        Returns:
        tuple: Disparity coordinates (disparityCoors) and disparity values (disparityPoints).
        """
        disparityPoints = []
        disparityCoors = []
        print(pointsL[53])
        for i,_ in enumerate(pointsL):
            disparityCoors.append(pointsL[i][0])
            disparityPoints.append(pointsL[i][0][0]-pointsR[i][0][0])
        return np.array(disparityCoors), np.array(disparityPoints)

    def reprojectPointsTo3D(self, disparityCoors, disparityValues):
        """
        Reprojects disparity points to 3D space using disparity values and the Q matrix.

        Parameters:
        disparityCoors (np.ndarray): Disparity coordinates.
        disparityValues (np.ndarray): Disparity values.

        Returns:
        tuple: 3D points (pcPoints) and colors (pcColors) of the point cloud.
        """
        pcPoints = []
        pcColors = []
        for di,d in enumerate(disparityCoors):
            x = d[0]
            y = d[1]
            px = -round((x - self.w/2)*(1/self.Q[3][2])/disparityValues[di],2)
            py = -round((y - self.h/2)*(-1/self.Q[3][2])/disparityValues[di],2)
            pz = -round((1/self.Q[3][2])*self.Q[2][3]/disparityValues[di],2)
            pcPoints.append([px,py,pz])
            pcColors.append([255,0,0])
        return np.array(pcPoints), np.array(pcColors)
    
    
    def generateCheckerboardTo3D(self, squaresSize = 8,boardSize = (9, 6),show = False):
        """
        Integrates chessboard corner detection, disparity computation, and 3D reprojection.

        Parameters:
        squaresSize (int): Size of the chessboard squares in mm.
        boardSize (tuple): Number of internal corners in (columns, rows).
        show (bool): Flag to display the detected corners.

        Returns:
        tuple: 3D points (pcPoints) and colors (pcColors) of the checkerboard.
        """
        pointsL, pointsR = self.checkerboardCorners(squaresSize,boardSize,show=show)
        disparityCoors, disparityValues = self.disparity(pointsL, pointsR)
        self.pcPoints, self.pcColors = self.reprojectPointsTo3D(disparityCoors, disparityValues)
        return self.pcPoints, self.pcColors
    
    def roiCrop(self):
        """
        Crops stereo images based on image points (ROI).

        Returns:
        tuple: Cropped left and right images (cropImgL, cropImgR) and bounding box (bbox).
        """
        border = 20
        yMin = np.min(self.imgPoints[0:2,1])
        yMax = np.max(self.imgPoints[0:2,1])
        bbox = [0,yMin-border,self.w,(yMax - yMin) + border*2]
        cropImgL = cropBBox(self.imgL, bbox)
        cropImgR = cropBBox(self.imgR, bbox)
        return cropImgL,cropImgR, bbox
    
    def cropMask(self):
        """
        Applies a mask to the left image based on the provided image mask (imgMask).
        """
        self.imgL = cv2.bitwise_and(self.imgL, self.imgL, mask=self.imgMask)

    def maskLine(self):
        """
        Generates a binary mask with a line based on image points (imgPoints).

        Returns:
        np.ndarray: Mask image with a line.
        """
        blackImage = np.zeros((self.h,self.w), dtype=np.uint8)
        maskImg = cv2.line(blackImage, self.imgPoints[0], self.imgPoints[1], 255, 10)
        return maskImg
    
    def maskPoints(self):
        """
        Generates binary masks with circles based on image points (imgPoints).

        Returns:
        list of np.ndarray: List of mask images with circles.
        """
        maskImgs = []
        for i, p in enumerate(self.imgPoints):
            blackImage = np.zeros((self.h,self.w), dtype=np.uint8)
            maskImg = cv2.circle(blackImage ,p, 10, 255, -1)
            maskImgs.append(maskImg)
        return maskImgs
    
    @staticmethod
    def gridPoints(points, depth):
        """
        Generate grid points around specified points and extract corresponding depth values.

        Parameters:
        points (list of tuples): List of center points (x, y) around which grid points will be generated.
        depth (np.ndarray): 2D array containing depth values.

        Returns:
        tuple: A tuple containing two lists:
            - listCoordinates (list of np.ndarray): List of coordinates for each grid point mask.
            - listValues (list of np.ndarray): List of corresponding depth values for each mask.
        """
        listCoordinates = []
        listValues = []
        # Generate a grid of points for the circle mask once
        radius = 10
        x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask_ = x**2 + y**2 <= radius**2
        maskCoordinates = np.column_stack(np.where(mask_))
        
        for point in points:
            center_x, center_y = point
            
            # Adjust coordinates to the actual center point
            coordinates = maskCoordinates.copy()
            coordinates[:, 0] += center_x - radius
            coordinates[:, 1] += center_y - radius

            # Extract depth values for the current mask
            maskedDepth = depth[coordinates[:, 1], coordinates[:, 0]]
            mask_ = maskedDepth > depth.min()
            coordinates = coordinates[mask_]
            maskedDepth = maskedDepth[mask_]
            listCoordinates.append(coordinates)
            listValues.append(maskedDepth)

        return listCoordinates, listValues

    def disparitySGBM(self):
        """
        Computes disparity using Semi-Global Block Matching (SGBM) algorithm.

        Returns:
        np.ndarray: Disparity map.
        """
        if self.imgL.ndim == 2:
            imgChannels = 1
        else:
            imgChannels = 3
        blockSize = 7
        param = {'minDisparity': self.minDisp,
                 'numDisparities': self.numDisp,
                 'blockSize': blockSize,
                 'P1': 8 * imgChannels * blockSize ** 2,
                 'P2': 32 * imgChannels * blockSize ** 2,
                 'disp12MaxDiff': 1, 
                 'preFilterCap': 63,
                 'uniquenessRatio': 12, 
                 'speckleWindowSize': 400, 
                 'speckleRange': 2,
                 'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                 }
        
        sgbm = cv2.StereoSGBM_create(**param)
        # self._cropMask()
        imageL = self.imgL
        imageR = self.imgR

        if self.imgPoints is not None:
            imageL, imageR, bbox = self.roiCrop()
        
        disparity = sgbm.compute(imageL, imageR)
        
        disparity = disparity.astype(np.float32) / 16
        if self.imgPoints is not None:
            minDisparity = np.min(disparity)
            blankDisparity = np.zeros((self.h,self.w), dtype=np.uint8)*minDisparity
            blankDisparity[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = disparity
            
            return blankDisparity
        
        return disparity
    
    def generatePointTo3D(self):
        """
        Integrates disparity computation, grid point generation, and 3D reprojection based on image points (imgPoints).

        Returns:
        np.ndarray: Mean 3D points between two sets of grid points.
        """
        depthImg = self.disparitySGBM()
        listCoordinates, listValues = self.gridPoints(self.imgPoints, depthImg)
        listPoints = []
        listColors = []
        for i, coors in enumerate(listCoordinates):
            self.pcPoints, self.pcColors = self.reprojectPointsTo3D(coors, listValues[i])
            self.outlierFilter()
            listPoints.append(self.pcPoints)
            listColors.append(self.pcColors)
        mStart = np.mean(listPoints[0], axis=0)
        mEnd = np.mean(listPoints[1], axis=0)
        self.pcColors = listColors[1]
        self.pcPoints = listPoints[1]
        return np.array([mStart, mEnd])
    
    def generateMaskTo3D(self):
        """
        Integrates disparity computation, masking, and 3D reprojection based on image mask (imgMask).

        Returns:
        np.ndarray: 3D points (pcPoints) and colors (pcColors) of the masked region.
        """
        depth = self.disparitySGBM()
        if self.imgMask is not None:
            depth = cv2.bitwise_and(depth, depth, mask = self.imgMask)
    
        #Q matrix is form during the calibration
        Q1 = np.float32([[1, 0, 0, -self.w / 2.0],
                        [0, -1, 0, self.h / 2.0],
                        [0, 0, 0, self.Q[2, 3]],
                        [0, 0, -self.Q[3, 2], self.Q[3, 3]]])
        

      
        pcPoints = cv2.reprojectImageTo3D(depth, Q1)
        pcColors = cv2.cvtColor(self.imgL, cv2.COLOR_BGR2RGB)
        mask_ = depth > depth.min()
        self.pcColors = pcColors[mask_]
        self.pcPoints = pcPoints[mask_]
        return np.array([self.pcPoints, self.pcColors])
    
    def generateImageTo3D(self):
        """
        Integrates disparity computation, masking, and 3D reprojection based on the full stereo images.

        Returns:
        tuple: 3D points (pcPoints) and colors (pcColors) of the full stereo images.
        """
        depthImg = self.disparitySGBM()
        if self.imgMask is not None:
            depthImg = cv2.bitwise_and(depthImg, depthImg, mask = self.imgMask)

        Q1 = np.float32([[1, 0, 0, -self.w / 2.0],
                        [0, -1, 0, self.h / 2.0],
                        [0, 0, 0, self.Q[2, 3]],
                        [0, 0, -self.Q[3, 2], self.Q[3, 3]]])
        points = cv2.reprojectImageTo3D(depthImg, Q1)
        colors = cv2.cvtColor(self.imgL, cv2.COLOR_BGR2RGB)
        mask_ = depthImg > depthImg.min()
        self.pcColors = colors[mask_]
        self.pcPoints = points[mask_]
        return self.pcPoints, self.pcColors 


    def outlierFilter(self):
        """
        Filters outlier points from the point cloud based on statistical and radius criteria.

        Returns:
        tuple: Filtered 3D points (pcPoints) and colors (pcColors) of the point cloud.
        """
        mean = np.mean(self.pcPoints, 0)
        idx = np.where(self.pcPoints[:,2] > mean[2]-5)[0]
        idx = np.asarray(idx, dtype=np.int32)
        pointRemain = self.pcPoints[idx]
        colorRemain = self.pcColors[idx]
        pcdMain = o3d.geometry.PointCloud()
        pcdMain.points= o3d.utility.Vector3dVector(pointRemain)
        pcdMain.colors= o3d.utility.Vector3dVector(colorRemain)
        cl, ind = pcdMain.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
        cl, ind = cl.remove_radius_outlier(nb_points=20, radius=1)   # It takes more time.
        self.pcPoints = np.asarray(cl.points)
        self.pcColors = np.asarray(cl.colors)
        return self.pcPoints, self.pcColors
    
    def show(self):
        """
        Visualizes the generated 3D point cloud using Open3D.
        """
        if self.pcPoints is not None and self.pcColors is not None:
            points = self.pcPoints.reshape(-1, 3)
            colors = self.pcColors.reshape(-1, 3) 
            colors = np.asarray(colors/255) # rescale to 0 to 1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])
        else:
            print("No 3D points or colors to visualize.")

    def write(self, pathName = 'pointcloud.ply'):
        """
        Saves the generated point cloud to a PLY file.

        Parameters:
        pathName (str): File path to save the PLY file.
        """
        plyHeader = '''ply
                    format ascii 1.0
                    element vertex %(vertNum)d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    '''
        points = self.pcPoints.reshape(-1, 3)
        colors = self.pcColors.reshape(-1, 3) 
        points = np.hstack([points, colors])
        with open(pathName, 'wb') as f:
            f.write((plyHeader % dict(vertNum=len(points))).encode('utf-8'))
            np.savetxt(f, points, fmt='%f %f %f %d %d %d ')


