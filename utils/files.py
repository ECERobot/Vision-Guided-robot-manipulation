import os
import math
import cv2
import numpy as np
import yaml

class Config:
    def __init__(self, yamlFile):
        self.yamlFile = yamlFile
        self.data = self._loadYaml()
        # Transformation matrix
        self.topRobot = self.data['topRobot']
        self.topTop = self.data['topTop']
        self.s0Top = self.data['s0Top']
        self.s120Top = self.data['s120Top']
        self.s240Top = self.data['s240Top']
        self.stationRobot = self.data['stationRobot']
        # Stereo camera config
        self.offsets = self.data['offsets']
        self.serialIDs = self.data['serialIDs']
        self.cropSize = self.data['cropSize']
        self.fullSize = self.data['fullSize']
        self.exposure = self.data['exposure']
        self.stereoMapFiles = self.data['stereoMapFiles']
        self.disparities = self.data['disparities']

        # Deep learning config
        self.aiCropSize = self.data['aiCropSize']
        self.weights = self.data['weights']
        self.modelTypes = self.data['modelTypes']
        self.modelPath = self.data['modelPath']

    def _loadYaml(self):
        with open(self.yamlFile, 'r') as file:
            return yaml.safe_load(file)

def loadStereoMaps(folderPath,cfg):
    Qs =[]
    stereoMaps = [] 
    cameraMatrixs = []

    stereoMapFiles = cfg.stereoMapFiles
    for fi, f in enumerate(stereoMapFiles):
        print('- ' + f)
        cvFile = cv2.FileStorage(folderPath + f, cv2.FileStorage_READ)
        stereoMapR_x = cvFile.getNode('stereoMapR_x').mat()
        stereoMapR_y = cvFile.getNode('stereoMapR_y').mat()
        stereoMapL_x = cvFile.getNode('stereoMapL_x').mat()
        stereoMapL_y = cvFile.getNode('stereoMapL_y').mat()

        cameraMatrixL = cvFile.getNode('camera_L_mxt').mat()
        cameraMatrixR = cvFile.getNode('camera_R_mxt').mat()
        Q = cvFile.getNode('q').mat()
        cvFile.release()
        stereoMaps.append((stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y))
        Qs.append(Q)
        cameraMatrixs.append((cameraMatrixL,cameraMatrixR))
    return stereoMaps, Qs, cameraMatrixs