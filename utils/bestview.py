import os
from ultralytics import YOLO
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon, Point
from utils.cvfunc import distance2D

def drawBudDir(image, startPs, dirPs, endPs, color=[0, 0, 255]):
    """
    Draws lines representing bud directions and optionally circles at the starting points on an image.

    Args:
        image (numpy.ndarray): The original image on which to draw.
        startPs (list of tuples): List of starting points for the bud directions.
        dirPs (list of tuples): List of direction points (not used in the current implementation).
        endPs (list of tuples): List of ending points for the bud directions.
        color (list, optional): Color of the lines and circles in BGR format. Default is [0, 0, 255] (red).

    Returns:
        numpy.ndarray: A copy of the input image with lines drawn for bud directions.
    """
    img = image.copy()
    for si, s in enumerate(startPs):
        if np.sum(startPs[si])!=0:
            cv2.line(img, startPs[si], endPs[si], color, thickness = 2)
            # cv2.circle(img,startPs[si], 3, (255, 255, 0), -1)
        else:
            pass
    return img

def showInfor(listImgs,imgNames, listBudDirs, candidateIDs, featureVal,filterID =1, color = [0,0,255]):
    """
    Combines multiple images and annotates them with information related to candidate views.

    Args:
        listImgs (list of numpy.ndarray): List of images to be displayed.
        imgNames (list): List of names or identifiers corresponding to each image.
        listBudDirs (list): List of bud direction data for each candidate view.
        candidateIDs (list): List of identifiers for each candidate view.
        featureVal (list): List of feature values to be displayed (e.g., confidence indices, lengths, etc.).
        filterID (int, optional): Identifier indicating which type of information to display. Default is 1.
        color (list, optional): Color used for annotations in BGR format. Default is [0, 0, 255] (red).

    Returns:
        numpy.ndarray: A merged image of all annotated images horizontally concatenated for display.
        list of numpy.ndarray: The individual annotated images.
    """
    imgs = listImgs.copy()
    for ci, c in enumerate(candidateIDs):
        budDirs = listBudDirs[ci]
        imgs[c] = drawBudDir(imgs[c], budDirs[0], budDirs[1], budDirs[2], color)
        if np.sum(budDirs[3])!=0:
            cv2.circle(imgs[c],budDirs[3], 4, (0, 255, 255), -1)
    s = ['MCI: ','DR: ', 'BN: ', 'LR: ', 'AI: ' ]
    coor = [[0, 20], [120, 20], [240, 20], [380, 20], [500, 20]]
    text = []
    for ci,c  in enumerate(candidateIDs):
        if(len(featureVal)==0):
            continue
        if(filterID==6):
            cv2.putText(imgs[c], 'Best-view', (0, 140), 0, 0.7, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            continue
        if(filterID==1):
            cv2.putText(imgs[c], 'Name: ' + imgNames[c], (0, 80), 0, 0.7, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
        if(filterID==2):
            cv2.putText(imgs[c], 'Best-view candidate', (0, 110), 0, 0.7, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
        if(filterID==-1):
            text = str(round(float(featureVal[ci]),2))
            cv2.putText(imgs[c], 'CI: ' + text, (0, 50), 0, 0.7, [50, 50, 50], thickness=2, lineType=cv2.LINE_AA)
        elif(filterID==-2):
            text = str(round(float(featureVal[ci]),2))
            cv2.putText(imgs[c], 'MI: ' + text, (120, 50), 0, 0.7, [50, 50, 50], thickness=2, lineType=cv2.LINE_AA)
        elif(filterID==3):
            text = str(int(featureVal[ci])) +'/' + str(round(np.sum(featureVal)/len(featureVal),1))
            cv2.putText(imgs[c], s[filterID-1]+text, coor[filterID-1], 0, 0.7, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
        else:
            if(len(featureVal) != len(candidateIDs)):
                continue
            if (filterID==1):
                featureVal[ci] = 0 if featureVal[ci] < 0 else featureVal[ci]
                featureVal[ci] = 1 if featureVal[ci] >1 else featureVal[ci]
            text = str(round(float(featureVal[ci]),2))
            cv2.putText(imgs[c], s[filterID-1]+text, coor[filterID-1], 0, 0.7, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
    merge_imgs = np.concatenate((imgs[0], imgs[1], imgs[2], imgs[3]), axis=1)  
    return merge_imgs, imgs


def pairAngle(v1, v2):
    """
    Calculate the absolute angle in degrees between two vectors v1 and v2.

    Args:
    v1 (numpy.ndarray): First vector.
    v2 (numpy.ndarray): Second vector.

    Returns:
    float: Absolute angle in degrees between v1 and v2.
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # To avoid numerical errors
    return abs(np.degrees(angle))  # Convert radians to degrees

class BestView:
    def __init__(self,images, imageNames, candidateIDs, budSegs, budDirs, cIndexs, syncIndexs, show = False):
        """
        Initializes the BestView object with input data.

        Args:
            images (list): List of images for processing.
            imageNames (list): Names or identifiers corresponding to each image.
            candidateIDs (list): Identifiers for each candidate view.
            budSegs (list): Segmentations of buds in each view.
            budDirs (list): Directions and properties of buds in each view.
            cIndexs (list): Confidence indices for each candidate view.
            syncIndexs (list): Synchronization indices for each candidate view.
            show (bool, optional): Flag indicating whether to show intermediate results. Defaults to False.
        """
        self.images = images
        self.imageNames = imageNames
        self.candidateIDs = candidateIDs
        self.budSegs = budSegs
        self.budDirs = budDirs
        self.cIndexs = cIndexs
        self.syncIndexs = syncIndexs
        self.show = show

    @staticmethod
    def clearIndexFilter(clIndex, clThreshhold = 0.1):
        """
        Static method to filter confidence indices based on a threshold.

        Args:
            clIndex (float): Confidence index value to be filtered.
            clThreshhold (float, optional): Threshold value for filtering. Defaults to 0.1.

        Returns:
            bool: True if the confidence index is above or equal to the threshold, False otherwise.
        """
        return clIndex >= clThreshhold

    def candidatesUpdate(self,filterIndexs):
        """
        Updates candidate views based on filter indices.

        Args:
            filterIndexs (list): List of boolean indices indicating which candidates to keep.

        Returns:
            None
        """
        budDirs_ = []
        budSegs_ = []
        candidateIDs_ = []
        for fi, f in enumerate(filterIndexs):
            _, _, _, centroid = self.budDirs[fi]
            if f and np.sum(centroid)!=0:
                budDirs_.append(self.budDirs[fi])
                budSegs_.append(self.budSegs[fi])
                candidateIDs_.append(self.candidateIDs[fi])
        self.budDirs = budDirs_ 
        self.budSegs = budSegs_ 
        self.candidateIDs = candidateIDs_ 

    def badBudRemoval(self, distThresh = 0.65):
        """
        Removes poorly defined buds based on distance ratios.

        Args:
            distThresh (float, optional): Threshold for distance ratio. Defaults to 0.65.

        Returns:
            list: List of distance ratios for each candidate view after removal.
        """
        distRatios = []
        budDirs_ = []
        budSegs_ = []
        for ci,c in enumerate(self.candidateIDs):

            sps, dps, eps, centroid = self.budDirs[ci]
            segMasks, segRois, segClassIDs, segScores = self.budSegs[ci]
            dists = []
            lengths = []
            for si, s in enumerate(sps):
                if np.sum(s) !=0:
                    dist = distance2D(s, centroid)
                    dists.append(dist)
                    length = distance2D(s, eps[si])
                    lengths.append(length)
                else:
                    dists.append(0)
            if len(dists) > 0:
                distID = dists.index(max(dists))
                distRatio = 1-(dists[distID]/np.mean(lengths))
                distRatio = 0 if distRatio <0 else distRatio
                distRatios.append(distRatio)
                if(distRatio<distThresh):
                    sps = np.delete(sps,distID,0)
                    dps = np.delete(dps,distID,0)
                    eps = np.delete(eps,distID,0)
                    segRois = np.delete(segRois,distID,0)
                    segClassIDs = np.delete(segClassIDs,distID,0)
                    segScores = np.delete(segScores,distID,0)
                    segMasks = np.delete(segMasks,distID,0)
            sps = list(map(np.array, sps))
            dps = list(map(np.array, dps))
            eps = list(map(np.array, eps))
            budDirs_.append(np.array([sps, dps, eps, centroid], dtype=object))
            budSegs_.append(np.array([segMasks, segRois, segClassIDs, segScores], dtype=object))
        self.budDirs = budDirs_
        self.budSegs = budSegs_
        return distRatios
    
    def budNumRatioFilter(self):
        """
        Filters candidate views based on the number of buds present.

        Returns:
            list: List of boolean values indicating which candidate views to keep.
            numpy.array: Array of integers representing the number of buds in each view.
        """
        budNums = []
        budNumFilter = []
        for i ,id in enumerate(self.candidateIDs):
            sps, dps, eps, centroid = self.budDirs[i]
            budNums.append(len(sps)-1)
        for bni, bn in enumerate(budNums):
            budNumFilter.append(True  if bn >=(np.sum(budNums)/len(self.candidateIDs)) else False)
        return budNumFilter, np.array(budNums).astype(int)
    
    def lengthRatioFilter(self):
        """
        Filters candidate views based on the ratio of bud lengths.

        Returns:
            list: List of boolean values indicating which candidate views to keep based on length ratios.
            list: List of length ratios for each candidate view.
            list: List of length candidates for further consideration.
        """
        totalLengths = []
        lengthRatios = []
        lengthFilter = []
        for i ,id in enumerate(self.candidateIDs):
            sps, dps, eps, centroid = self.budDirs[i]
            lengths = []
            for si, s in enumerate(sps):
                if np.sum(s)==0:
                    continue
                length = distance2D(s, eps[si])
                lengths.append(length)
            totalLengths.append(np.sum(lengths))
        for li, l in enumerate(totalLengths):
            lengthRatios.append(round(l/np.sum(totalLengths),3)) 
            lengthFilter.append(False)
        lengthID = np.argsort(lengthRatios)[::-1]

        for idi, id in enumerate(lengthID):
            if idi <2:
                lengthFilter[id] = True
            else:
                lengthFilter[id] = False
        lengthCandidates = []
        for fi, f in enumerate(lengthFilter):
            if f:
                lengthCandidates.append(lengthRatios[fi])
        return lengthFilter, lengthRatios,lengthCandidates
    
    def angleIndexFilter(self, lengthCandidates):
        """
        Filters candidate views based on angle indices between buds.

        Args:
            lengthCandidates (list): List of candidates based on length ratios.

        Returns:
            list: List of boolean values indicating which candidate views to keep based on angle indices.
            list: List of angle indices for each candidate view.
        """
        angleIndexs = []
        angleFilter = []
        for lbi, lb in enumerate(self.budDirs):
            sps, dps, eps, centroid = lb
            if not len(sps)>2:
                continue
            angles = []
            for si, s in enumerate(sps):
                if(np.sum(s)==0):
                    continue
                angles_ = []
                for si_, s_ in enumerate(sps):
                    if(np.sum(s_)==0):
                        continue
                    if si == si_ or (si_-si)<0:
                        continue
                    v1 = eps[si] - s
                    v2 = eps[si_] - s_
                    angles_.append(pairAngle(v1, v2))
                if len(angles_)==0:
                    continue
                angles.append(min(angles_))
            angleIndexs.append(min(angles))
        if len(angleIndexs)==1:
            angleFilter.append(True)
        elif len(angleIndexs)==2:
            angleID = angleIndexs.index(max(angleIndexs))
            sAngleID = 0 if (angleID +1) ==2 else 1
            angleBias = 1.2
            if angleIndexs[angleID] <= angleIndexs[sAngleID]*angleBias:
                lengthID = lengthCandidates.index(max(lengthCandidates))
                for ci, c in enumerate(self.candidateIDs):
                    if lengthID == ci:
                        angleFilter.append(True)
                    else:
                        angleFilter.append(False)
            else:
                for ci, c in enumerate(self.candidateIDs):
                    if angleID == ci:
                        angleFilter.append(True)
                    else:
                        angleFilter.append(False)
        else:
            pass
        return angleFilter, angleIndexs
  
    def run(self):
        """
        Executes the entire filtering pipeline and returns the best candidate view.

        Returns:
            tuple: Tuple containing the best candidate ID, segmentation data, direction data,
                and optionally intermediate images if `show` is True.
        """
        try:
            error = False
            isClears = []
            syncClearIndexs = []
            for i in range(0,len(self.cIndexs)):
                syncClearIndex = self.cIndexs[i] + self.syncIndexs[i]
                syncClearIndexs.append(syncClearIndex)
                isClear = self.clearIndexFilter(syncClearIndex, clThreshhold = 0.2)
                isClears.append(isClear)
            if self.show:
                _, images_ = showInfor(self.images, self.imageNames, self.budDirs, self.candidateIDs, featureVal=self.cIndexs, filterID=-1, color = [255,0,0])
                _, images_ = showInfor(images_, self.imageNames, self.budDirs, self.candidateIDs,featureVal=self.syncIndexs, filterID=-2, color = [255,0,0])
                _, images_ = showInfor(images_, self.imageNames, self.budDirs, self.candidateIDs,featureVal=syncClearIndexs, filterID=1, color = [255,0,0])
            self.candidatesUpdate(isClears)
            distRatios = self.badBudRemoval()
            if self.show:
                _,images_ = showInfor(images_, self.imageNames, self.budDirs, self.candidateIDs,featureVal=distRatios, filterID=2)
            budNumFilter, budNums = self.budNumRatioFilter()
            if self.show:
                _,images_ = showInfor(images_, self.imageNames, self.budDirs, self.candidateIDs,featureVal=budNums, filterID=3)
            self.candidatesUpdate(budNumFilter)
            lengthFilter, lengthRatios,lengthCandidates = self.lengthRatioFilter()
            if self.show:
                _,images_ = showInfor(images_, self.imageNames, self.budDirs, self.candidateIDs,featureVal=lengthRatios, filterID=4)
            self.candidatesUpdate(lengthFilter)
            angleFilter, angleIndexs = self. angleIndexFilter(lengthCandidates)
            if self.show:
                _,images_ = showInfor(images_, self.imageNames, self.budDirs, self.candidateIDs,featureVal=angleIndexs, filterID=5)
            self.candidatesUpdate(angleFilter)
            if self.show:
                _,images_ = showInfor(images_, self.imageNames, self.budDirs, self.candidateIDs,featureVal=angleIndexs, filterID=6)
                return self.candidateIDs[0],self.budSegs[0],self.budDirs[0], images_, error
            return self.candidateIDs[0], self.budSegs[0], self.budDirs[0], self.images, error
        except:
            error = True
            return None, None, None, None, error