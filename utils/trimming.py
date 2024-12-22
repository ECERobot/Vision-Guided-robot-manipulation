import os
import numpy as np
from pypylon import genicam, pylon
from utils.cvfunc import distance2D, linearInterpolate,contourInfor


class Trimming:
    def __init__(self, image, orchidSeg, budSegs, budDirs, trimRatio= 0.7, show = False): 
        self.image = image
        self.orchidSeg = orchidSeg
        self.budSegs = budSegs
        self.budDirs = budDirs
        self.trimRatio = trimRatio
        self.show = show

    def darkenLength(self, segMask, startP, dirP):
        """
        Calculate the end point for the length of a darken part.

        Inputs:
        - segMask: Segmentation mask (darken mask).
        - startP: Starting point.
        - dirP: Direction point.

        Outputs:
        - Returns the end point.
        """
        _, cnt, _, _ = contourInfor(self.image, segMask)

        # Vectorized projection calculation
        line_vec = dirP - startP
        line_len = np.dot(line_vec, line_vec)
        line_unitvec = line_vec / np.sqrt(line_len)
        
        points = np.array(cnt).reshape(-1, 2)
        point_vecs = points - startP
        point_vecs_scaled = point_vecs / np.sqrt(line_len)
        ts = np.dot(point_vecs_scaled, line_unitvec)
        projections = startP + np.outer(ts, line_vec)
        
        # Distance calculation using vectorized operations
        dists = np.linalg.norm(projections - startP, axis=1)

        # Find the point with the maximum distance
        id = np.argmax(dists)
        endP = projections[id]
        return np.array(endP).astype(int)

    def run(self):
        sps, dps, eps, centroid = self.budDirs
        segMasks, segRois, segCls, segScores = self.budSegs

        dirPoints = []
        trimPoints = []
        clsIDs = []  # 0: trim leaf, 1: trim darken
        error = False
        for sp, dp, ep in zip(sps,dps,eps):
            if np.mean(sp) == 0:
                continue
            d = distance2D(sp, dp)
            d1 = distance2D(sp, ep)
            ratio = d/d1
            if ratio >= 0.5 and d> 100:
                d = 100

            trimPoint = linearInterpolate(sp, dp, d*self.trimRatio)
            clsID = 0
            clsIDs.append(clsID)
            trimPoints.append(trimPoint)
            dirPoints.append(sp)
        dirDarken = np.array(np.mean(trimPoints, axis=0)).astype(int)
        if not np.isnan(dirDarken).any():
            darkID = np.array(np.where(segCls==1)).reshape(-1)
            
            if len(darkID) !=0:
                endDarken = self.darkenLength(segMasks[darkID[0]], dirDarken, centroid)
                trimDarken = linearInterpolate(endDarken,dirDarken, 20)
                clsID = 1
                clsIDs.insert(0, clsID)
                trimPoints.insert(0, trimDarken)
                dirPoints.insert( 0,  dirDarken)
            clsIDs =np.asarray(clsIDs).astype(int).tolist()
            trimPoints = np.asarray(trimPoints).astype(int).tolist()
            dirPoints =np.asarray(dirPoints).astype(int).tolist()

            return np.array([clsIDs,dirPoints,trimPoints], dtype=object), error

        else:
            error = True
            return None, error
            
          