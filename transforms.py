
import numpy as np
import math
from numpy.linalg import inv
def list2array(points):
    points = np.array([list(elem) for elem in points])
    return points

def euler2Rotation(posture):
    """
    Convert Euler angles and position into a 4x4 rotation matrix.
    
    Parameters:
    posture (list): A list of six values [x, y, z, alpha, beta, gamma].
                    x, y, z: Position coordinates.
                    alpha, beta, gamma: Euler angles in degrees.
    
    Returns:
    np.ndarray: A 4x4 rotation matrix representing the transformation.
    """
    x, y, z, alpha, beta, gamma = posture
    alpha = alpha * math.pi / 180
    beta = beta * math.pi / 180
    gamma = gamma * math.pi / 180

    Rx = np.array([[1, 0, 0, 0],
                    [0, math.cos(alpha), -math.sin(alpha), 0],
                    [0, math.sin(alpha), math.cos(alpha), 0],
                    [0, 0, 0, 1]])

    Ry = np.array([[math.cos(beta), 0, math.sin(beta), 0],
                    [0, 1, 0, 0],
                    [-math.sin(beta), 0, math.cos(beta), 0],
                    [0, 0, 0, 1]])

    Rz = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                    [math.sin(gamma), math.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Combine rotation matrices and include translation
    R = np.dot(Rz, np.dot(Ry, Rx))
    R[0, 3] = x 
    R[1, 3] = y
    R[2, 3] = z

    return R

def rotation2Euler(R):
    """
    Convert a 4x4 rotation matrix to Euler angles and position.
    
    Parameters:
    R (np.ndarray): A 4x4 rotation matrix.
    
    Returns:
    np.ndarray: Euler angles and position [x, y, z, alpha, beta, gamma].
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    x = R[0, 3]
    y = R[1, 3]
    z = R[2, 3]

    if not singular:
        Rx = math.atan2(R[2, 1], R[2, 2])
        Ry = math.atan2(-R[2, 0], sy)
        Rz = math.atan2(R[1, 0], R[0, 0])
    else:
        Rx = math.atan2(-R[1, 2], R[1, 1])
        Ry = math.atan2(-R[2, 0], sy)
        Rz = 0

    # Convert angles to degrees
    Rx = Rx * 180 / math.pi
    Ry = Ry * 180 / math.pi
    Rz = Rz * 180 / math.pi

    return np.array([round(x, 2), round(y, 2), round(z+10, 2), round(Rx, 2), round(Ry, 2), round(Rz, 2)])

    
def rigidTransform3D(A, B):
    """
    Compute the 3D rigid transformation (rotation and translation) that aligns points A to points B.
    
    Parameters:
    A (np.ndarray): 3xN matrix of points (N points in 3D).
    B (np.ndarray): 3xN matrix of points (N points in 3D), where A and B have the same shape.
    
    Returns:
    np.ndarray: 3x3 rotation matrix R.
    np.ndarray: 3x1 translation vector t.
    """
    "Matrices A and B must have the same shape."
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # Find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # Ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # Subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # Compute covariance matrix H
    H = Am @ np.transpose(Bm)

    # Perform Singular Value Decomposition (SVD) on H
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix R
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    # Compute translation vector t
    t = -R @ centroid_A + centroid_B

    return R, t

def transform(points, transformMatrix):
    """
    Apply a transformation matrix to a set of 3D points.
    
    Parameters:
    points (np.ndarray): An Nx3 array of points, where each row is a point [x, y, z].
    transformMatrix (np.ndarray): A 4x4 transformation matrix.
    
    Returns:
    np.ndarray: An Nx3 array of transformed points.
    """
    # Convert points to homogeneous coordinates (Nx4)
    ones = np.ones((points.shape[0], 1))
    pointsHomogeneous = np.hstack((points, ones))

    # Transform the points using the transformation matrix
    transformedPointsHomogeneous = np.array(transformMatrix) @ pointsHomogeneous.T

    # Convert back to 3D coordinates (remove the homogeneous component)
    transformedPoints = transformedPointsHomogeneous[:3].T
    return transformedPoints

def transformCamera2Camera(points, candidateID, cfg):
    """
    Transform points from one camera coordinate system to another based on candidateID.
    
    Parameters:
    points (np.ndarray): An Nx3 array of points, where each row is a point [x, y, z].
    candidateID (int): Identifier for the target camera coordinate system.
    cfg (object): Configuration object containing transformation matrices.
    
    Returns:
    np.ndarray: An Nx3 array of points transformed to the target camera coordinate system.
    """
    if candidateID ==0:
        topPoints = transform(points, cfg.topTop)
    elif candidateID ==1:
        topPoints = transform(points, cfg.s0Top)
    elif candidateID ==2:
        topPoints = transform(points, cfg.s120Top)
    elif candidateID ==3:
        topPoints = transform(points, cfg.s240Top)
    else:
        raise ValueError("Invalid candidateID")
    return topPoints


def transformCamera2Robot(points, transformMatrix):
    """
    Transform camera points to robot points using a transformation matrix.
    
    Parameters:
    points (np.ndarray): An Nx3 array of points, where each row is a point [x, y, z].
    transformMatrix (np.ndarray): A 4x4 transformation matrix.
    
    Returns:
    np.ndarray: An Nx3 array of points transformed to the robot coordinate system.
    """
    robotPoints = transform(points, transformMatrix)
    return robotPoints
  
def transformRobot2Station(points, transformMatrix):
    """
    Transform robot points to station points using a transformation matrix.
    
    Parameters:
    points (np.ndarray): An Nx3 array of points, where each row is a point [x, y, z].
    transformMatrix (np.ndarray): A 4x4 transformation matrix.
    
    Returns:
    np.ndarray: An Nx3 array of points transformed to the station coordinate system.
    """

    stationPoints = transform(points, inv(transformMatrix))
    return stationPoints

def transformStation2Robot(points, transformMatrix):
    """
    Transform station points to robot points using a transformation matrix.
    
    Parameters:
    points (np.ndarray): An Nx3 array of points, where each row is a point [x, y, z].
    transformMatrix (np.ndarray): A 4x4 transformation matrix.
    
    Returns:
    np.ndarray: An Nx3 array of points transformed to the robot coordinate system.
    """
    robotPoints = transform(points, transformMatrix)
    return robotPoints

def transformStation(points, rotateAngle):
    """
    Rotate station points by a given angle.
    
    Parameters:
    points (np.ndarray): An Nx3 array of points, where each row is a point [x, y, z].
    rotateAngle (float): The rotation angle in radians.
    
    Returns:
    np.ndarray: An Nx3 array of rotated points.
    """
    rotateMat = euler2Rotation([0,0,0,0,0,rotateAngle])
    stationPoints = transform(points, rotateMat)
    return stationPoints

def translate(posture, translate):
    """
    Apply a translation to a posture described by [x, y, z, alpha, beta, gamma].
    
    Parameters:
    posture (list or array): A list or array of six values [x, y, z, alpha, beta, gamma].
                              The first three are the position coordinates.
                              The last three are the Euler angles in degree.
    translate (list or array): A list or array of three translation values [dx, dy, dz].
    
    Returns:
    list: A list of six values [new_x, new_y, new_z, new_alpha, new_beta, new_gamma].
          The first three are the new position coordinates.
          The last three are the new Euler angles in degree after applying the translation.
    """
    postureMat = euler2Rotation(posture)
    transfromMat = np.eye(4)
    transfromMat[:3, 3] = translate
    postureMat = np.matmul(postureMat, transfromMat)
    postureEul = rotation2Euler(postureMat)

    return postureEul


def pixel2World(points, Q, imgz = (2048,2592), depth = 480):
    wPoints = []
    for p in points:
        x, y = p
        cx = imgz[1]/2
        cy = imgz[0]/2
        f = Q[2][3]
        px = -(depth*(x-cx)/f)
        py = (depth*(y-cy)/f)
        pz = -depth
        wPoints.append([round(px,2), round(py,2), round(pz,2)])
    return np.array(wPoints).tolist()

def groupPixel2World(groupPoints, Q, imgz = (2048,2592), depth = 480):
    wGroupPoints = []
    for gi, group in enumerate(groupPoints):
        clsIDs, dirPoints, trimPoints = group
        wDirPoints = pixel2World(dirPoints, Q, imgz=imgz, depth=depth)
        wTrimPoints = pixel2World(trimPoints, Q, imgz=imgz, depth=depth)
        wGroupPoint = np.array([clsIDs,wDirPoints,wTrimPoints], dtype=object)
        wGroupPoints.append(wGroupPoint)
    return wGroupPoints
    
def groupTransformCamera2Robot(groupPoints, transformMatrix):
    rGroupPoints = []
    for gi, group in enumerate(groupPoints):
        clsIDs, dirPoints, trimPoints = group
        dirPoints= list2array(dirPoints)
        trimPoints= list2array(trimPoints)
       
        rDirPoints = transformCamera2Robot(dirPoints, transformMatrix).tolist()
        rTrimPoints = transformCamera2Robot(trimPoints, transformMatrix).tolist()
        rGroupPoint = np.array([clsIDs,rDirPoints,rTrimPoints], dtype=object)
        rGroupPoints.append(rGroupPoint)
    return rGroupPoints

def groupTransformRobot2Station(groupPoints, transformMatrix):
    sGroupPoints = []
    for gi, group in enumerate(groupPoints):
        clsIDs, dirPoints, trimPoints = group
        dirPoints= list2array(dirPoints)
        trimPoints= list2array(trimPoints)
        sDirPoints = transformRobot2Station(dirPoints, transformMatrix).tolist()
        sTrimPoints = transformRobot2Station(trimPoints, transformMatrix).tolist()
        sGroupPoint = np.array([clsIDs,sDirPoints,sTrimPoints], dtype=object)
        sGroupPoints.append(sGroupPoint)
    return sGroupPoints

def groupTransformStation2Robot(groupPoints, transformMatrix):
    rGroupPoints = []
    for gi, group in enumerate(groupPoints):
        clsIDs, dirPoints, trimPoints = group
        dirPoints= list2array(dirPoints)
        trimPoints= list2array(trimPoints)
        rDirPoints = transformStation2Robot(dirPoints, transformMatrix).tolist()
        rTrimPoints = transformStation2Robot(trimPoints, transformMatrix).tolist()
        rGroupPoint = np.array([clsIDs,rDirPoints,rTrimPoints], dtype=object)
        rGroupPoints.append(rGroupPoint)
    return rGroupPoints

def groupTransformStation(groupPoints, rotateAngle):
    rGroupPoints = []
    for gi, group in enumerate(groupPoints):
        clsIDs, dirPoints, trimPoints = group
        dirPoints= list2array(dirPoints)
        trimPoints= list2array(trimPoints)
        rDirPoints = transformStation(dirPoints, rotateAngle).tolist()
        rTrimPoints = transformStation(trimPoints, rotateAngle).tolist()
        rGroupPoint = np.array([clsIDs,rDirPoints,rTrimPoints], dtype=object)
        rGroupPoints.append(rGroupPoint)
    return rGroupPoints



def contourTransform(points, angle, center, isContour = False):

    if isContour:
        points = points.reshape(-1, 2)
    # Convert angle to radians
    angleRad = np.deg2rad(angle)
    
    # Create the rotation matrix
    cosA = np.cos(angleRad)
    sinA = np.sin(angleRad)
    R = np.array([[cosA, - sinA], [sinA, cosA]])
    
    # Translate points to origin (center point)
    translatedPoints = points - center
    
    # Apply the rotation matrix
    rotatedPoints = np.dot(translatedPoints, R.T)
    
    # Translate points back to the original position
    rotatedPoints += center
    if isContour:
        rotatedPoints = rotatedPoints.reshape(-1, 1, 2)
    
    return rotatedPoints