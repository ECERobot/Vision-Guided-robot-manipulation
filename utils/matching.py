import os
import cv2 as cv
import numpy as np
from utils.cvfunc import*
def list2array(points):
    points = np.array([list(elem) for elem in points])
    return points
def dist2D(p1, p2):
    # dist = math.hypot(p1[1] - p1[0], p2[1] - p2[0])
    dist = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    return int(dist)

def add_bg(img,cnt):
    overMask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    cv2.drawContours(overMask, cnt, -1, (255, 255, 255), -1, cv2.LINE_AA)
    kernel = np.ones((5, 5), np.uint8)
    overMask = cv2.dilate(overMask, kernel, iterations=3)
    dst = cv2.bitwise_and(img, img, mask=overMask)
    bg = np.ones_like(dst, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=overMask)
    full_image = bg+ dst
    return full_image

def remove_bg(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ksize = (3, 3)
    image = cv2.blur(img_gray, ksize) 
    ret, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
    kernel1 = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=3)
    thresh = cv2.dilate(thresh, kernel, iterations=5)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    img_iso = add_bg(img,contours)
    return img_iso

def cropTemplate(srcP, totalP, srcImg):
    srcImg = remove_bg(srcImg)
    # cv2.imshow("srcImg 1", resized_img(srcImg, 50))
    # cv2.waitKey(0)
    avgP = np.int32(np.mean(totalP,axis=0))
    # print('avgP: ', avgP)
    maxDist = 0
    for (_, ins) in enumerate(totalP):
        dist = dist2D(avgP, ins)
        # dist = math.dist(avgP, ins)
        if dist > maxDist:
            maxDist = dist
            # print('maxDist: ', maxDist)
    cropPar= np.int32([maxDist*2, maxDist*2])
    # print('cropPar: ', cropPar)
    originP = np.int32(avgP - maxDist*1)
    localP = (srcP - originP)
    dstImag = srcImg[originP[1]:originP[1]+cropPar[1], originP[0]:originP[0]+cropPar[0]]
    return localP, dstImag

def cropTemplate(srcImgs, groupTrimPoints, aiGroupBBoxes):
    localGroupPoints =[]
    cropPars = []
    templateImgs =[]
    for i, img in enumerate(srcImgs):
        h, w = img.shape[:2]
        clsIDs, dirPoints, trimPoints = groupTrimPoints[i]
        print("clsIDs: ",clsIDs)
        if len(clsIDs)<1:
            continue
        avgPoint = np.int32(np.mean(list2array(trimPoints),axis=0))
        maxDist = 0
        for pi, p in enumerate(list2array(trimPoints)):
            dist = dist2D(avgPoint, p)
            # dist = math.dist(avgP, ins)
            if dist > maxDist:
                maxDist = dist
        overMask = np.zeros((h, w,1), np.uint8)
        overMask = cv2.circle(overMask ,avgPoint, maxDist, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=overMask)
        bg = np.ones_like(dst, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=overMask)
        changedImg = bg + dst
        cropPar= np.int32([maxDist*2, maxDist*2])
        cropPars.append(cropPar)
        originP = np.int32(avgPoint - maxDist*1)
        dstImag = changedImg[originP[1]:originP[1]+cropPar[1], originP[0]:originP[0]+cropPar[0]]
        templateImgs.append(dstImag)
        dirPoints_ = (list2array(dirPoints)-originP).tolist()
        trimPoints_ = (list2array(trimPoints) - originP).tolist()
        localGroupPoint = np.array([clsIDs,dirPoints_,trimPoints_], dtype=object)
        localGroupPoints.append(localGroupPoint)
    return templateImgs, localGroupPoints, cropPars


def tempMatching(srcImg, srcTempImg, localGroupPoints):
    MIN_MATCH_COUNT = 7
    # srcImg = remove_bg(srcImg)
    # cv2.imshow("srcImg 2", resized_img(srcImg, 50))
    # cv2.waitKey(0)
    tempImg= cv2.cvtColor(srcTempImg, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    imgRGB = srcImg.copy()
    (h, w) = imgRGB.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(tempImg,None)
    kp2, des2 = sift.detectAndCompute(img,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)
    print("num of feature: ", len(good))
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        rigidM, rigid_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        M1, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        M1[2][0] = 0
        M1[2][1] = 0
        M2 = M1
        M2[0][0] = rigidM[0][0]
        M2[0][1] = rigidM[0][1]
        M2[0][2] = rigidM[0][2]
        M2[1][0] = rigidM[1][0]
        M2[1][1] = rigidM[1][1]
        M2[1][2] = rigidM[1][2]

        matchesMask = mask.ravel().tolist()
        h,w = tempImg.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M2)
        
        # points = np.array(localP, "int32")
        # points = np.expand_dims(points, axis=0)
        # points_ = cv2.transform(points, M2).astype("int32")
        # points_ = points_.squeeze()
        # globalP = np.delete(points_, np.s_[2], 1)
       
        # imgRGB = cv.polylines(imgRGB,[np.int32(dst)],True,255,3, cv.LINE_AA)
        # cv.circle(imgRGB, np.int32(dst[0][0]), radius=5, color=(0, 0, 255), thickness=-1)
        # cv.circle(imgRGB, np.int32(dst[3][0]), radius=5, color=(0,255, 0), thickness=-1)
        # for (_,ins) in enumerate(globalP):
        #     cv.circle(imgRGB, np.int32(ins), radius=7, color=(0,0, 255), thickness=-1)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
        dstImg = cv.drawMatches(srcTempImg,kp1,imgRGB,kp2,good,None,**draw_params)
        cv.imshow("dstImg",dstImg)
        # return globalP, dstImg, 0
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        # return localP, imgRGB, 1
    
