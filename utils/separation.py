import os
from ultralytics import YOLO
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon, Point
import math
from utils.cvfunc import distance2D


class Separation:
    def __init__(self, image, budSeg, budDir):
        self.image = image
        self.budSeg = budSeg
        self.budDir = budDir

    # def angleEstimation(self)
    @staticmethod
    def dist2D(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @staticmethod
    def midPoint(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        return np.array([xm, ym]).astype(int)

    @staticmethod
    def angle2D(v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # To avoid numerical errors
        return abs(np.degrees(angle))
    
    @staticmethod
    def AngleCal(p1, p2):
        deltaY = p1[1] - p2[1]
        deltaX = p1[0] - p2[0]
        angleInDegrees = math.atan2(deltaY, deltaX) * 180 / 3.14
        if angleInDegrees<0:
            return angleInDegrees+360
        else:
            return angleInDegrees
        
    @staticmethod   
    def extendLine(p1, p2, distance=10):
        diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        p3_x = int(p1[0] + distance*np.cos(diff))
        p3_y = int(p1[1] + distance*np.sin(diff))
        p4_x = int(p1[0] - distance*np.cos(diff))
        p4_y = int(p1[1] - distance*np.sin(diff))
        return (p3_x, p3_y), (p4_x, p4_y)
    
    def budLength(self):
        segMasks, segRois, segCls, segScores = self.budSeg
        sps, dps, eps, centroid = self.budDir
        budLengths = []
        for si, s in enumerate(sps):
            budLengths.append(self.dist2D(s, eps[si]))
        return budLengths
    
    def singleBudAngle(self, tempDirs = None):
        sps, dps, eps, centroid = self.budDir
        angles = []
        if tempDirs is not None:
            for tdi, td in enumerate(tempDirs):
                angle = self.AngleCal(centroid, td)
                angles.append(angle)
        else:
            for si, s in enumerate(sps):
                angle = self.AngleCal(s, eps[si])
                angles.append(angle)
        return np.array(angles)
    
    def budLengthSorted(self, budLengths):
        segMasks, segRois, segCls, segScores = self.budSeg
        sps, dps, eps, centroid = self.budDir
        ids = np.array(budLengths).argsort()#[::-1]
        sps = np.array(sps)[ids]
        dps = np.array(dps)[ids]
        eps = np.array(eps)[ids]
        segMasks = np.array(segMasks)[ids]
        segRois = np.array(segRois)[ids]
        segCls = np.array(segCls)[ids]
        segScores = np.array(segScores)[ids]
        budLengths = np.array(budLengths)[ids]
        self.budSeg = np.array([segMasks, segRois, segCls, segScores], dtype=object)
        self.budDir = np.array([sps, dps, eps, centroid], dtype=object)
        return budLengths

    def budAngleSorted(self, angles, budLengths, tempBudDir = None):
        segMasks, segRois, segCls, segScores = self.budSeg
        sps, dps, eps, centroid = self.budDir
        ids = np.array(angles).argsort()
        angles = np.array(angles)[ids]
        sps = np.array(sps)[ids]
        dps = np.array(dps)[ids]
        eps = np.array(eps)[ids]
        segMasks = np.array(segMasks)[ids]
        segRois = np.array(segRois)[ids]
        segCls = np.array(segCls)[ids]
        segScores = np.array(segScores)[ids]
        budLengths = np.array(budLengths)[ids]
        if tempBudDir is not None:
            tempBudDir = np.array(tempBudDir)[ids]
        self.budSeg = np.array([segMasks, segRois, segCls, segScores], dtype=object)
        self.budDir = np.array([sps, dps, eps, centroid], dtype=object)
        return angles, budLengths, tempBudDir
    
    def contourInfor(self, mask, erode = False):
        h, w = self.image.shape[:2]
        mCntImg = np.zeros((h, w, 1), dtype=np.uint8)
        mask = np.squeeze(mask)
        mask = (mask*255).astype(np.uint8)
        if erode:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None and len(contours) > 0:
            mCnt = max(contours, key = cv2.contourArea)
            cv2.drawContours(mCntImg, [mCnt],  0, 255, -1)
            M = cv2.moments(mCnt)
            centroidX = int(M['m10'] / (M['m00']+1e-10))
            centroidY = int(M['m01'] / (M['m00']+1e-10))
            return mCntImg, mCnt,np.array([centroidX,centroidY]).astype(int)
        else:
            return None, None, None
        
    @staticmethod
    def maxContour(binaryImage, BBox = False):
        contours,_ = cv2.findContours(binaryImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        mCnt = max(contours, key = cv2.contourArea)
        M = cv2.moments(mCnt)
        cX = int(M['m10'] / (M['m00']+1e-10))
        cY = int(M['m01'] / (M['m00']+1e-10))
        if BBox:
            x, y, w, h = cv2.boundingRect(mCnt)
            p1 = [x,y]
            p2 = [x+w,y]
            p3 = [x+w,y+h]
            p4 = [x,y+h]
            return mCnt, [p1,p2,p3,p4] , np.array([cX,cY]).astype(int)
        else:
            return mCnt, np.array([cX,cY]).astype(int)
    
    @staticmethod
    def normalize(v):
        norm=np.linalg.norm(v)
        if norm==0:
            return v
        return v/norm
    
    @staticmethod
    def findClosestPoints(cnts):
        cnt1, cnt2 = cnts
        # Expand dimensions of contour1 for broadcasting
        cnt1Expanded = np.expand_dims(cnt1, axis=1)

        # Compute the distance between each pair of points in the contours
        distances = np.linalg.norm(cnt1Expanded - cnt2, axis=-1)

        # Find the indices of the minimum distances
        minIndices = np.unravel_index(distances.argmin(), distances.shape)

        # Retrieve the closest points from the contours
        closestPointCnt1 = cnt1[minIndices[0]]
        closestPointCnt2 = cnt2[minIndices[1]]

        return np.array(closestPointCnt1).reshape(-1).astype(int), np.array(closestPointCnt2).reshape(-1).astype(int)

    @staticmethod
    def closestPointOnVector(A, B, P):
        # Vector AB
        AB = B - A
        
        # Vector AP
        AP = P - A
        normalizedAB = AB / np.linalg.norm(AB)
        projectionLength = np.dot(AP, normalizedAB)
        projectionVector = normalizedAB * projectionLength

        # Closest point on the vector
        closestPoint = A + projectionVector
        return np.array(closestPoint).astype(int)
    
    def drawSeparationLine(self, sp, ep, color = [0,255,255]):
        if len(sp) !=0 and len(ep)!=0:
            sp1, sp2 = self.extendLine(sp, ep, distance=100)
            try:
                cv2.line(self.image, sp1, sp2, color, 2)
            except:
                print(sp1, sp2)
  
    @staticmethod
    def rotatePoint(cxy, angle, p):
        s = np.sin(np.radians(angle))
        c = np.cos(np.radians(angle))

        # Translate point back to origin
        px,py = p
        px -= cxy[0]
        py -= cxy[1]

        # Rotate point
        xnew = px * c - py * s
        ynew = px * s + py * c

        # Translate point back
        px = xnew + cxy[0]
        py = ynew + cxy[1]

        return np.array([px, py]).astype(int)
    
    
    @staticmethod
    def isAngleBetweenClosestRange(target, angle1, angle2):
        """
        Determine if the target angle is between angle1 and angle2 in the closest range.
        
        Parameters:
        target (float): The target angle to check.
        angle1 (float): The first boundary angle.
        angle2 (float): The second boundary angle.
        
        Returns:
        bool: True if target is between angle1 and angle2 in the closest range, False otherwise.
        """
        # Normalize angles to be in the range [0, 360)
        target = target % 360
        angle1 = angle1 % 360
        angle2 = angle2 % 360
        
        # Compute the direct distances
        directDist = (angle2 - angle1) % 360
        reverseDist = (angle1 - angle2) % 360
        
        # Determine the shortest range
        if directDist < reverseDist:
            startAngle = angle1
            endAngle = angle2
        else:
            startAngle = angle2
            endAngle = angle1
        
        # Check if target is in the range [start_angle, end_angle]
        if startAngle <= endAngle:
            return startAngle <= target <= endAngle
        else:
            return startAngle <= target or target <= endAngle

    @staticmethod   
    def findFarthestPoint(contour, refPoint):
        """
        Find the farthest point on a contour from a reference point.
        
        Parameters:
            contour : numpy.ndarray
                The contour points.
            ref_point : tuple
                The reference point (x, y).
            
        Returns:
            farthest_point : tuple
                The farthest point on the contour from the reference point.
            max_distance : float
                The distance to the farthest point.
        """
        maxDistance = 0
        farthestPoint = None
        
        for point in contour:
            # Get the coordinates of the contour point
            x, y = point
            
            # Calculate the Euclidean distance to the reference point
            distance = np.sqrt((x - refPoint[0])**2 + (y - refPoint[1])**2)
            
            # Update the farthest point if the distance is greater
            if distance > maxDistance:
                maxDistance = distance
                farthestPoint = (x, y)
        
        return np.array(farthestPoint).astype(int)

    @staticmethod
    def rotateImage(image, angle):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotatedImage = cv2.warpAffine(image, M, (w, h))
        return rotatedImage
    
    def rotateBBox(self, cntImg, id):
        sps, dps, eps, centroid = self.budDir
        h, w = self.image.shape[:2]
        angle = self.AngleCal(sps[id], dps[id])
        rotImg = self.rotateImage(cntImg, (angle-90))
        cnt, bbox,_ = self.maxContour(rotImg, BBox = True)
        cxy = [w/2,h/2]
        rotBBox = []
        for p in bbox:
            rotBBox.append(self.rotatePoint(cxy, (angle-90), p))
        return np.array(rotBBox).astype(int)

    def cropContour(self, pairID, show =False):
        segMasks, segRois, segCls, segScores = self.budSeg
        sps, dps, eps, centroid = self.budDir
        h, w = self.image.shape[:2]
        mCnts = []
        bottomBBoxs = []
        cXYs = []
        for id in pairID:
            blankImg = np.zeros((h, w, 1), dtype=np.uint8)
            mCntImg, mCnt, mCntCentroid = self.contourInfor(segMasks[id], erode = True)
            r =  self.dist2D(centroid, dps[id])
            blankImg = cv2.circle(blankImg ,centroid, int(r*0.8), (255, 255, 0), -1)
            andImg = cv2.bitwise_and(blankImg, mCntImg)
            mCnt,cXY = self.maxContour(andImg)
            cXYs.append(cXY)
            # cv2.drawContours(self.image, [mCnt],  0, 255, 1)
            mCnts.append(mCnt)
            rotBBox = self.rotateBBox(andImg, id)
            r_ = self.dist2D(cXY,self.midPoint(rotBBox[2],rotBBox[3]))
            bottomBBoxs.append([rotBBox[2],rotBBox[3]])
            # cv2.drawContours(self.image, [rotBBox], 0, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.circle(self.image ,cXY, int(r_), (0, 0, 255), 1)
        cPC1, cPC2 = self.findClosestPoints(mCnts)
        if show:
            cv2.circle(self.image ,cPC1, 3, (100, 100, 255), -1)
            cv2.circle(self.image ,cPC2, 3, (100, 100, 255), -1)
        midP = self.midPoint(cPC1, cPC2)
        meanCXYs = np.mean(cXYs,axis=0).astype(int)
        # cv2.circle(self.image ,meanCXYs, 5, (255, 255, 255), -1)
        return cPC1, cPC2, bottomBBoxs, centroid, midP
    
    def cropAllContour(self, pairID):
        segMasks, segRois, segCls, segScores = self.budSeg
        sps, dps, eps, centroid = self.budDir
        h, w = self.image.shape[:2]
        mCnts = []
        bottomBBoxs = []
        cXYs = []
        tempDirs = []
        for id in pairID:
            blankImg = np.zeros((h, w, 1), dtype=np.uint8)
            mCntImg, mCnt, mCntCentroid = self.contourInfor(segMasks[id], erode = True)
            r =  self.dist2D(centroid, dps[id])
            blankImg = cv2.circle(blankImg ,centroid, int(r*0.8), (255, 255, 0), -1)
            andImg = cv2.bitwise_and(blankImg, mCntImg)
            mCnt, cXY = self.maxContour(andImg)
            cXYs.append(cXY)
            # cv2.drawContours(self.image, [mCnt],  0, 255, 1)
            mCnts.append(mCnt)
            rotBBox = self.rotateBBox(andImg, id)
            r_ = self.dist2D(cXY,self.midPoint(rotBBox[2],rotBBox[3]))
            bottomBBoxs.append([rotBBox[2],rotBBox[3]])
            # cv2.drawContours(self.image, [rotBBox], 0, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.circle(self.image ,cXY, int(r_), (0, 0, 255), 1)
        tempDirs.append(centroid)
        for ci, c in enumerate(cXYs):
            tempDirs.append(c)
            # cv2.line(self.image, centroid, c, (255,255,255), thickness = 1)
        # cv2.circle(self.image ,meanCXYs, 5, (255, 255, 255), -1)
        return tempDirs
        
    def twoBudsAngle(self):
        sps, dps, eps, centroid = self.budDir
        v1 = eps[1] - sps[1]
        v2 = eps[2] - sps[2]
        pairAngle = self.angle2D(v1, v2)
        return round(pairAngle,1)
    
    def threeBudsAngle(self):
        sps, dps, eps, centroid = self.budDir
        pairAngles = []
        minPairIDs = []
        maxPairIDs = []
        for i in range(1,len(sps)):
            v1 = eps[i] - sps[i]
            pairAngles_ = []
            for j in range(1,len(sps)): 
                if i == j:
                    continue
                v2 = eps[j] - sps[j]
                pairAngles_.append(self.angle2D(v1, v2))
            pairAngles.append(np.sum(pairAngles_))
        ID = pairAngles.index(max(pairAngles))
        minPairIDs.append(ID+1)
        maxPairIDs.append(ID+1)
        angles = []
        angleIDs = []
        for i in range(1,len(sps)):
            if i == ID+1:
                continue
            v1 = eps[ID+1] - sps[ID+1]
            v2 = eps[i] - sps[i]
            angles.append(self.angle2D(v1, v2))
            angleIDs.append(i)
        minID = angles.index(min(angles))
        maxID = angles.index(max(angles))
        minPairIDs.append(angleIDs[minID])
        maxPairIDs.append(angleIDs[maxID])
        minPairAngle = angles[minID]
        maxPairAngle = angles[maxID]
        return [round(minPairAngle,1),round(maxPairAngle,1)], [minPairIDs,maxPairIDs]
    
    def pairBudsAngleTest(self):
        sps, dps, eps, centroid = self.budDir
        pairAngles = []
        pairAngleIDs = []
        sumAngleIDs = []
        for i in range(1,len(sps)):
            pairAngles_ = []
            pairAngleIDs_ = []
            for j in range(1,len(sps)):
                if i == j :
                    continue
                dupI = []
                dupJ = []
                if len(pairAngleIDs)>0:
                    dupI = np.array(np.where(np.array(pairAngleIDs)[:,0]==j)).reshape(-1)
                    dupJ = np.array(np.where(np.array(pairAngleIDs)[:,1]==i)).reshape(-1)
                if len(dupI)!=0 and len(dupJ)!=0:
                   continue
                v1 = eps[i] - sps[i]
                v2 = eps[j] - sps[j]
                pairAngles_.append(self.angle2D(v1, v2))
                pairAngleIDs_.append([i,j])
            if(len(pairAngles_)==0):
                continue
            minID = pairAngles_.index(min(pairAngles_))
            pairAngles.append(pairAngles_[minID])
            pairAngleIDs.append(pairAngleIDs_[minID])
            sumAngleIDs.append(np.sum(pairAngleIDs_[minID]))
    
    def biSector(self,separateP, pairID):
        sps, dps, eps, centroid = self.budDir
        v1 = eps[pairID[0]] - sps[pairID[0]]
        v2 = eps[pairID[1]] - sps[pairID[1]]
        v1 = self.normalize(v1)
        v2 = self.normalize(v2)
        v3 = self.normalize(v1+v2)
        epS =  np.array(v3*100 + separateP).astype(int)
        reCentroid = self.closestPointOnVector(separateP, epS, centroid)
        return np.array(reCentroid).astype(int)

    def mutiBudsAngle(self,budAngles):
        sps, dps, eps, centroid = self.budDir
        pairAngles = []
        pairIDs = []
        targetPairIDs = []
        for i in range(1,len(budAngles)):
            j=i+1
            if j == len(budAngles):
                j = 1
            pairIDs.append([i,j])
            pairAngle = budAngles[j] - budAngles[i]
            if pairAngle < 0:
                pairAngle+=360
            pairAngles.append(pairAngle)
        maxID = pairAngles.index(max(pairAngles))
        # draw=============================================================================================
        # for pi, p in enumerate(pairIDs):
        #     text = '[{},{}]: {}'.format(p[0],p[1],int(pairAngles[pi]))
        #     cv2.putText(self.image, text, (0, 170+30*pi), 0, 0.6, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
        #==================================================================================================
        targetPairIDs.append(pairIDs[maxID])
        targetAngle = budAngles[maxID+1]+(pairAngles[maxID]/2)
        targetAngle = targetAngle if targetAngle<360 else targetAngle - 360
        sysTargetAngle = targetAngle+180
        sysTargetAngle = sysTargetAngle if sysTargetAngle<360 else sysTargetAngle - 360
        pairIDs = np.delete(pairIDs,maxID,0)
        for idi, id in enumerate(pairIDs):
            isContain = self.isAngleBetweenClosestRange(sysTargetAngle, budAngles[id[0]], budAngles[id[1]])
            if isContain:
                targetPairIDs.append(id)
        # print("targetPairIDs: ",targetPairIDs)
        return targetPairIDs

    def run(self):
        try:
            error = False
            budL = self.budLength()
            budA = self.singleBudAngle()
            budA_, budL_,_ = self.budAngleSorted(budA,budL)
            # print("budA_: ",budA_)
            startP = []
            endP = []
            if len(budA)-1 ==3: # case 1: 3 buds
                pairAngle, pairIDs = self.threeBudsAngle()
                _, _,_, cenP, sepP = self.cropContour(pairIDs[0],show =False)
                if(pairAngle[1]>100):
                    _, _,bottomBBoxs, _, _ = self.cropContour(pairIDs[1])
                    farPoint1 = self.findFarthestPoint(bottomBBoxs[0], sepP)
                    farPoint2 = self.findFarthestPoint(bottomBBoxs[1], sepP)
                    startP = self.midPoint(farPoint1, farPoint2)
                    # startP = cenP
                    endP = sepP
                    startP = self.closestPointOnVector(startP, endP, cenP)
                else:
                    startP = cenP
                    endP = sepP
                    # angle = self.AngleCal(startP, endP)
                    # isContain = self.isAngleBetweenClosestRange(angle, budA_[pairIDs[0][0]], budA_[pairIDs[0][1]])
                    # if not isContain:
                    startP = self.biSector(endP, pairIDs[0])
            elif len(budA)-1 ==2:# case 2: 2 buds
                pairAngle = self.twoBudsAngle()
                if pairAngle>=90:
                    _,_,_,startP,endP = self.cropContour([1,2])
                    startP = self.biSector(endP, [1,2])
            elif len(budA)-1 > 3:
                pairIDs = []
                for bi, b in enumerate(budA):
                    if bi == 0:
                        continue
                    pairIDs.append(bi)
                tempDirs = self.cropAllContour(pairIDs)
                tempA = self.singleBudAngle(tempDirs)
                tempA_, tempL_, tempBudDir = self.budAngleSorted(tempA,budL_,tempDirs)
                tempA__ = self.singleBudAngle()
                

                targetPairIDs = self.mutiBudsAngle(tempA__)

                _, _,_, cenP, sepP1 = self.cropContour(targetPairIDs[1],show =False)
                _, _,_, cenP, sepP2 = self.cropContour(targetPairIDs[0],show =False)
                endP = sepP1
                startP = self.closestPointOnVector(sepP2, endP, cenP)
                sps_, dps_, eps_, centroid_ = self.budDir
                # for ai, a in enumerate(tempA__):
                #     if ai ==0:
                #         continue
                #     cv2.putText(self.image, str(int(a)), eps_[ai], 0,0.6, [0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
                # for dpi, dp in enumerate(tempBudDir):
                #     if dpi ==0:
                #         continue
                #     cv2.putText(self.image, str(dpi), dp, 0,0.6, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            else:
                pass
            # if(len(startP)>0):
            #     cv2.circle(self.image ,startP, 5, (255, 255, 255), -1)
            self.drawSeparationLine(startP, endP)
            separatedLine = np.array([startP, endP]).astype(int)
            sps_, dps_, eps_, centroid_ = self.budDir
            cv2.circle(self.image ,centroid_, 4, (255, 0, 0), -1)
            try:
                cv2.circle(self.image ,startP, 4, (255, 0, 255), -1)
            except:
                error = True
                print('error')
                return self.image, None, error
            
            
            return self.image, separatedLine, error
        except:
            error = True
            return self.image,None, error
