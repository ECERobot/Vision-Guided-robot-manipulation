import os
import math
import numpy as np
import cv2

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

def cvShow(imgs):
    """
    Concatenate multiple images either horizontally or in a 2x2 grid.

    Args:
        imgs (list): List of images to be concatenated. Should contain 4 or 8 images.

    Returns:
        numpy.ndarray: Concatenated image grid.
    """
    if len(imgs) == 8:
        merge_img1 = np.concatenate((imgs[0], imgs[4], imgs[1], imgs[5]), axis=1)
        merge_img2 = np.concatenate((imgs[2], imgs[6], imgs[3], imgs[7]), axis=1)
        merge_img = np.concatenate((merge_img1, merge_img2), axis=0)
    else:
        merge_img = np.concatenate(imgs, axis=1)
    return merge_img

def cropImages(images, offsets, dropSize=(1600, 1600)):
    """
    Crop regions of interest from multiple images based on given offsets.

    Args:
        images (list): List of input images to be cropped.
        offsets (list): List of (offX, offY) tuples specifying crop offsets for each image.
        dropSize (tuple, optional): Size of the cropped region (width, height). Default is (1600, 1600).

    Returns:
        list: List of cropped images.
    """
    dh, dw = dropSize
    cropImgs = []
    for i,img in enumerate(images):
        offX, offY = offsets[i]
        cropImg = img[offY:offY + dh, offX:offX + dw]
        cropImgs.append(cropImg)
    return cropImgs

def findMaxContour(image, iter=5):
    """
    Find the largest contour in a binary image after performing erosion.

    Args:
        image (numpy.ndarray): Input binary image to find contours in.
        iter (int, optional): Number of erosion iterations. Default is 5.

    Returns:
        numpy.ndarray: Largest contour found in the image as an array of points.
    """
    kernel = np.ones((3, 3), np.uint8)
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(cv2.erode(thresh, kernel, iterations=iter), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None and len(contours) > 0:
        mCon = np.array(max(contours, key=cv2.contourArea))  
        return mCon 
    else:
        return None
    
def maxContour(mask):
    mask = np.squeeze(mask)
    mask = (mask*255).astype(np.uint8)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None and len(contours) > 0:
        mCnt = max(contours, key = cv2.contourArea)
        bbox = cv2.boundingRect(mCnt)
        M = cv2.moments(mCnt)
        centroid_x = int(M['m10'] / (M['m00']+1e-10))
        centroid_y = int(M['m01'] / (M['m00']+1e-10))
        return mCnt, bbox, np.array([centroid_x,centroid_y]).astype(int)
    else:
        return None, None, None
    
def contourInfor(image, mask, dilate =False):
    """
    Extract information from a mask applied to an image, including the masked image, contour, bounding box, and centroid.

    Args:
        image (numpy.ndarray): Input image on which the mask is applied.
        mask (numpy.ndarray): Binary mask to apply on the image.
        dilate (bool, optional): Whether to dilate the mask before processing. Default is False.

    Returns:
        tuple: Tuple containing:
            - numpy.ndarray: Masked image after applying the contour.
            - numpy.ndarray: Contour of the largest area in the mask.
            - tuple: Bounding box coordinates (x, y, width, height) of the contour.
            - tuple: Centroid coordinates (centroid_x, centroid_y) of the contour.
    """
    h, w = image.shape[:2]
    mask = np.squeeze(mask)
    mask = (mask*255).astype(np.uint8)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None and len(contours) > 0:
        black_image = np.zeros((h, w, 1), dtype=np.uint8)
        mCnt = max(contours, key = cv2.contourArea)
        maskImg = cv2.drawContours(black_image, [mCnt], -1, 255, cv2.FILLED)
        if dilate:
            kernel = np.ones((3, 3), np.uint8)
            maskImg = cv2.dilate(maskImg, kernel, iterations=3) 
        bbox = cv2.boundingRect(mCnt)
        M = cv2.moments(mCnt)
        centroid_x = int(M['m10'] / (M['m00']+1e-10))
        centroid_y = int(M['m01'] / (M['m00']+1e-10))
        return maskImg, mCnt, bbox, (centroid_x,centroid_y)
    else:
        return None, None, None, None
    

def closestPointContour(contour, targetPoint):
    """
    Find the closest point in a contour to a given target point.

    Args:
        contour (numpy.ndarray): Contour represented as an array of points.
        targetPoint (numpy.ndarray): Target point to find the closest point in the contour.

    Returns:
        tuple: Coordinates of the closest point in the contour.
    """
    minDistance = float('inf')
    closestPoint = None
    for point in contour:
        distance = np.linalg.norm(targetPoint - point)
        if distance < minDistance:
            minDistance = distance
            closestPoint = point
    return closestPoint[0]

def projectPoint2Line(point, linePoint1, linePoint2):
    """
    Project a point onto a line defined by two other points.

    Args:
        point (numpy.ndarray): Point to be projected onto the line.
        linePoint1 (numpy.ndarray): First point on the line.
        linePoint2 (numpy.ndarray): Second point on the line.

    Returns:
        numpy.ndarray: Projected point on the line as an integer numpy array [projectedX, projectedY].
    """
    point = np.array(point)
    linePoint1 = np.array(linePoint1)
    linePoint2 = np.array(linePoint2)
    lineDir = linePoint2 - linePoint1
    pointVec = point - linePoint1
    projectionScalar = np.dot(pointVec, lineDir) / np.dot(lineDir, lineDir)
    projectedPoint = linePoint1 + projectionScalar * lineDir
    return np.array(projectedPoint).astype(int)
    
def linearExtrapolation(p1, p2, dist):
    """
    Perform linear extrapolation of a point p2 based on a distance and direction from p1.

    Args:
        p1 (tuple): Coordinates of the first point (x1, y1).
        p2 (tuple): Coordinates of the second point (x2, y2).
        dist (float): Distance to extrapolate from p2.

    Returns:
        numpy.ndarray: Extrapolated coordinates as an integer numpy array [extrapolatedX, extrapolatedY].
    """
    x1, y1 = p1
    x2, y2 = p2
    # Calculate the angle and distance between the two points
    angle = math.atan2(y2 - y1, x2 - x1)
    # total_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # Calculate the extrapolated coordinates
    extrapolatedX = x2 + dist * math.cos(angle)
    extrapolatedY = y2 + dist * math.sin(angle)
    exP = np.array([extrapolatedX,extrapolatedY]).astype(int)
    return exP

def cropBBox(image, bbox):
    """
    Crop a region of interest from an image based on a bounding box.

    Args:
        image (numpy.ndarray): Input image from which to crop.
        bbox (tuple): Bounding box coordinates (x, y, width, height).

    Returns:
        numpy.ndarray: Cropped region of interest as a numpy array.
    """
    x, y, w, h = bbox
    cropObject = image[y:y+h, x:x+w]
    return cropObject

def midPoint2D(point1, point2):
    """
    Calculate the midpoint between two 2D points.

    Args:
        point1 (tuple): Coordinates of the first point (x1, y1).
        point2 (tuple): Coordinates of the second point (x2, y2).

    Returns:
        list: Coordinates of the midpoint [x_mid, y_mid].
    """
    x_mid = (point1[0] + point2[0]) / 2
    y_mid = (point1[1] + point2[1]) / 2
    return [x_mid, y_mid]

    
def linearInterpolate(p1, p2, distance):
    # Convert points to numpy arrays for easier computation
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Compute the total distance between the points
    totalDistance = np.linalg.norm(p2 - p1)
    
    # Compute the interpolation factor
    t = distance / totalDistance
    
    # Interpolate the point
    p = p1 + t * (p2 - p1)
    
    return np.asarray(p,dtype=int)


def distance2D(point1, point2):
    """
    Calculate the Euclidean distance between two 2D points.

    Args:
        point1 (tuple): Coordinates of the first point (x1, y1).
        point2 (tuple): Coordinates of the second point (x2, y2).

    Returns:
        float: Euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def destroyWindowProperty(window_name):
    try:
        # Attempt to get the window property to check if the window exists
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error as e:
        # Handle the error if the window does not exist
        print(f"Window '{window_name}' does not exist or cannot be accessed.")


def dePointsAiCrop(points, aiBBoxes,camOffsets, candidateID, aiCropSize=(640,640)):
    """
    Scale and reposition a set of points relative to a bounding box and camera offsets.

    Args:
        points (list): List of points to be scaled and repositioned.
        aiBBoxes (list): List of AI bounding boxes, where each entry is a tuple (aiX, aiY, aiH, aiW).
        camOffsets (list): List of camera offsets corresponding to each bounding box.
        candidateID (int): Index of the candidate bounding box and offsets to use.
        aiCropSize (tuple, optional): Size of the AI cropped region (width, height). Default is (640, 640).

    Returns:
        numpy.ndarray: Array of scaled and repositioned points as integers.
    """
    aiX,aiY, aiH, aiW = aiBBoxes[candidateID]
    offX, offY = camOffsets[candidateID]
    scaleFactor = aiH/aiCropSize[0]
    rePoints = []
    for i, p in enumerate(points):
        reP1 = p*scaleFactor
        reP2 = [reP1[0]+aiX, reP1[1]+aiY]
        reP3 = [reP2[0]+offX, reP2[1]+offY]
        rePoints.append(reP3)
    return np.array(rePoints).astype(int)

def dePointsAiGroupCrop(groupPoints, aiGroupBBoxes, camOffsets = (0,0), aiCropSize=(640,640)):
   
    
    offX, offY = camOffsets
    
    deGroups = []
   
    for gi, g in enumerate(groupPoints):
        clsIDs, dirPoints, trimPoints = g
        aiX,aiY, aiH, aiW = aiGroupBBoxes[gi]
        scaleFactor = aiH/aiCropSize[0]
        deDps = []
        deTps = []
        for dp, tp in zip(dirPoints, trimPoints):
            deDp1 = np.array(dp)*scaleFactor
            deTp1 = np.array(tp)*scaleFactor

            deDp2 = [deDp1[0]+aiX, deDp1[1]+aiY]
            deTp2 = [deTp1[0]+aiX, deTp1[1]+aiY]
            
            deDp3 = [deDp2[0]+offX, deDp2[1]+offY]
            deTp3 = [deTp2[0]+offX, deTp2[1]+offY]
            deDps.append(np.array(deDp3).astype(int))
            deTps.append(np.array(deTp3).astype(int))
        deGroup = np.array([clsIDs,deDps,deTps], dtype=object)
        deGroups.append(deGroup)
     
    return deGroups

def deMaskAiCrop(camOffsets, masks, candidateID, fullFrameSize = (2592, 2048), offsetSize = (1600,1600)):
    """
    Resize and position a mask onto a full-sized frame based on camera offsets.

    Args:
        masks (list): List of masks to be resized and positioned.
        camOffsets (list): List of camera offsets corresponding to each mask.
        candidateID (int): Index of the candidate mask and offsets to use.
        fullFrameSize (tuple, optional): Full size of the frame (width, height). Default is (2592, 2048).
        offsetSize (tuple, optional): Size of the offset frame (width, height). Default is (1600, 1600).

    Returns:
        numpy.ndarray: Full-sized frame image with the mask applied.
    """
    offX, offY = camOffsets[candidateID]
    mask = cv2.resize(masks[candidateID],offsetSize)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    mCnt = max(contours, key = cv2.contourArea)
    offsetImage = np.zeros((offsetSize[1],offsetSize[0],1), dtype=np.uint8)
    maskImg = cv2.drawContours(offsetImage, [mCnt], -1, 255, cv2.FILLED)
    fullFrameImage = np.zeros((fullFrameSize[1],fullFrameSize[0],1), dtype=np.uint8)

    fullFrameImage[offY:offY+offsetSize[1],offX:offX+offsetSize[0]] = maskImg

    return fullFrameImage


def changeBackground(image, maskSize, cnt):
    h, w = image.shape[:2]
    overMask = np.zeros((maskSize[0],maskSize[1],1), np.uint8)
    # print("Lenght  of countour",  cnt)
    # cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    # hull= cv2.convexHull(cnt)
    cv2.drawContours(overMask, [cnt], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # cv2.drawContours(overMask, [hull], -1, (255, 255, 255), -1, cv2.LINE_AA)
    overMask = cv2.resize(overMask, (h, w) )
    # cv2.imshow("Mask Bud",overMask)
    kernel = np.ones((3, 3), np.uint8)
    overMask = cv2.dilate(overMask, kernel, iterations=3)
    dst = cv2.bitwise_and(image, image, mask=overMask)
    bg = np.ones_like(dst, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=overMask)
    changedImg = bg + dst
    return changedImg

def perpendicularArrow(p1, p2, length=50):
    # Convert points to numpy arrays for easier computation
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Compute the direction vector
    directionVector = p2 - p1
    
    # Compute a perpendicular vector
    perpendicularVector1 = np.array([-directionVector[1], directionVector[0]])
    perpendicularVector2 = np.array([directionVector[1], -directionVector[0]])
    
    # Normalize the perpendicular vector
    perpendicularVector1 = perpendicularVector1 / np.linalg.norm(perpendicularVector1)
    perpendicularVector2 = perpendicularVector2 / np.linalg.norm(perpendicularVector2)
    
    # Scale the perpendicular vector to the desired length
    perpendicularVector1 = perpendicularVector1 * length
    perpendicularVector2 = perpendicularVector2 * length

    perPoint1 = (int(p2[0] + perpendicularVector1[0]), int(p2[1] + perpendicularVector1[1]))
    perPoint2 = (int(p2[0] + perpendicularVector2[0]), int(p2[1] + perpendicularVector2[1]))
    
    return perPoint1, perPoint2