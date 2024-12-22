import os
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import time
from shapely.geometry import Polygon, Point
from concurrent.futures import ThreadPoolExecutor
from utils.aug_plot import aug_plot
from utils.cvfunc import contourInfor, cropBBox,findMaxContour,maxContour, distance2D, closestPointContour, midPoint2D,resizeImage,linearExtrapolation, changeBackground
from ultralytics import YOLO
from ultralytics.engine.results import Results


def loadModels(path, weightNames, modelTypes):
    """
    Load multiple YOLO models with specified weights and types.

    Parameters:
    path (str): The base path where the model weight files are located.
    weightNames (list of str): A list of filenames for the model weights.
    modelTypes (list of str): A list of types for each model.

    Returns:
    list: A list of loaded YOLO models.
    """
    models = []
    for i, mn in enumerate(weightNames):
        print('- ' + mn)
        modePath = ''.join([path, mn])
        model = YOLO(modePath,task = modelTypes[i]) 
        models.append(model)
    return models

def toNumby(results):
    """
    Convert YOLO prediction results to NumPy arrays for masks, class IDs, scores, and bounding boxes.

    Parameters:
    results (list): A list of prediction results from YOLO models.

    Returns:
    np.ndarray: An array containing masks, rois, class IDs, and scores.
    """
    masks = []
    classIDs = []
    scores = []
    rois = []
    for i in range(len(results)):
        if results[i].masks is None:
            continue
        mask = results[i].masks.data.cpu().numpy()
        mask = (mask*255).astype(np.uint8)
        mask = np.squeeze(mask)
        classID = int((results.boxes.data[i][5]).cpu().numpy())
        score = (results.boxes.data[i][4]).cpu().numpy()*100
        roi = (results.boxes.data[i][:4]).cpu().numpy()
        masks.append(mask)  
        scores.append(score)  
        classIDs.append(3) 
        rois.append(roi)  
    return np.array([masks, rois, classIDs, scores], dtype=object)

def toBinary(masks):
    """
    Convert masks to binary format.

    Parameters:
    masks (list of np.ndarray): A list of masks to be converted to binary format.

    Returns:
    list of np.ndarray: A list of binary masks.
    """
    binaryMasks = []
    for mask in masks:
        binaryMask = np.where(mask > 1, 1, 0).astype(np.uint8)
        binaryMasks.append(binaryMask)
    return binaryMasks

def maskIOU(predMask1, predMask2):
    """
    Compute the Intersection over Union (IoU) between two masks.

    Parameters:
    predMask1 (np.ndarray): The first mask.
    predMask2 (np.ndarray): The second mask.

    Returns:
    torch.Tensor: The IoU between the two masks.
    """
    predMask1 = np.squeeze(predMask1)
    predMask1= predMask1[:, :, np.newaxis]
    predMask2 = np.squeeze(predMask2)
    predMask2= predMask2[:, :, np.newaxis]
    mask1 =torch.Tensor(predMask1) 
    mask2 =torch.Tensor(predMask2)
    H, W, N = mask1.shape
    H, W, M = mask2.shape
    mask1 = mask1.view(N, H*W)
    mask2 = mask2.view(M, H*W)
    intersection = torch.matmul(mask1, mask2.t())
    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection
    ret = torch.where(
        union == 0,
        torch.tensor(0., device=mask1.device),
        intersection / union,)
    return ret[0][0]

def nonMaxSuppression(predicted, iouThresh = 0.5):
    """
    Perform Non-Maximum Suppression (NMS) on predicted masks.

    Parameters:
    predicted (np.ndarray): An array containing masks, rois, class IDs, and scores.
    iouThresh (float): The IoU threshold for suppression. Default is 0.5.

    Returns:
    np.ndarray: Indices of the masks to keep.
    """
    masks, rois, classIDs, scores = predicted
    masks = toBinary(masks)
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        # Pick top mask and add its index to the list
        i = ixs[0]
        pick.append(i)
        IoUs = []
        for ix in ixs[1:]:
            IoUs.append(maskIOU(masks[i],masks[ix]))
            # print(mask_iou(masks[i],masks[ix]))
        # Identify masks with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(np.asarray(IoUs) > iouThresh)[0] + 1
        # Remove indicies of the picked and overlapped masks.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

def duplicateRemoval(predicted, iouThresh=0.5):
    """
    Remove duplicate predictions using Non-Maximum Suppression (NMS).

    Parameters:
    predicted (np.ndarray): An array containing masks, rois, class IDs, and scores.
    iouThresh (float): The IoU threshold for suppression. Default is 0.5.

    Returns:
    list: A list of filtered predictions.
    """
    keepInd = nonMaxSuppression(predicted, iouThresh)
    nmsResult = []
    for i in range(0,len(predicted)):
        nmsResult.append(predicted[i][keepInd])
    return nmsResult

def xyxy2xywh(bbox):
    """
    Convert bounding box from (x1, y1, x2, y2) format to (x, y, w, h) format.

    Parameters:
    bbox (np.ndarray): The bounding box in (x1, y1, x2, y2) format.

    Returns:
    np.ndarray: The bounding box in (x, y, w, h) format.
    """
    x,y = bbox[0:2]
    w,h  = bbox[2:4] - bbox[0:2]
    return np.array([x,y,w,h]).astype(int)


def aiGroupCrop(model, image, imgz = (640,640), conf = 0.5):
    """
    Crop the image using AI model predictions.

    Parameters:
    model: The YOLO model used for prediction.
    image (np.ndarray): The input image.
    imgz (tuple): The target size for the cropped image. Default is (640, 640).
    conf (float): The confidence threshold for predictions. Default is 0.5.

    Returns:
    tuple: The cropped image, offset, mask, and error flag.
    """

    h, w = image.shape[:2]
    segPred = model.predict(source = image, stream = False, conf = conf, save_txt  = False)[0]
    segPred = toNumby(segPred)
    segPred = duplicateRemoval(segPred,0.3)
    bgImages =[]
    offsets = []
    for i , mask in enumerate(segPred[0]):
        mCnt, _,_  = maxContour(mask)
        bgImage = changeBackground(image,imgz,mCnt)
        bbox = xyxy2xywh(segPred[1][i])

        maxSize = max(bbox[2], bbox[3]) + 120

        if maxSize < imgz[0]:
            maxSize = imgz[0]
        cx, cy = np.array([bbox[0] + bbox[2] / 2,bbox[1] + bbox[3] / 2]).astype(int)
        if (cy + int(maxSize/2))> h:
            cy = h - int(maxSize/2)
        elif cy-int(maxSize/2)<0:
            cy = int(maxSize/2)  

        if (cx + int(maxSize/2))> w:
            cx = w - int(maxSize/2)
        elif cx-int(maxSize/2)<0:
            cx = int(maxSize/2)  

        cropImg = bgImage[int(cy-(maxSize/2)):int(cy+(maxSize/2)), int(cx-(maxSize/2)):int(cx+maxSize/2)]

        offset = np.array([(cx-(maxSize/2)), (cy-(maxSize/2)), maxSize, maxSize]).astype(int) # x,y,w,h
        
        cropImg = cv2.resize(cropImg, imgz, interpolation=cv2.INTER_AREA)
        bgImages.append(cropImg)
        offsets.append(offset)
    
    return bgImages, np.array(offsets).astype(int)

   



def aiCrop(model, image, imgz = (640,640), conf = 0.5):
    """
    Crop the image using AI model predictions.

    Parameters:
    model: The YOLO model used for prediction.
    image (np.ndarray): The input image.
    imgz (tuple): The target size for the cropped image. Default is (640, 640).
    conf (float): The confidence threshold for predictions. Default is 0.5.

    Returns:
    tuple: The cropped image, offset, mask, and error flag.
    """
    try:
        h, w = image.shape[:2]
        segPred = model.predict(source = image, stream=False, conf = conf, save_txt  = False)[0]
        segPred = toNumby(segPred)
        segPred = duplicateRemoval(segPred,0.3)
        
        error = False
        if len(segPred[1]) != 0:
            bbox = xyxy2xywh(segPred[1][0])
        else:
            print('Not detect')
            bbox = np.array([w/2, h/2, w/2, h/2])
            error = True
            return cv2.resize(image, imgz, interpolation=cv2.INTER_AREA), None, None ,error
        mask = np.squeeze(segPred[0][0])
        mask = (mask*255).astype(np.uint8)
        mask = np.asarray(mask).reshape(imgz[1], imgz[0])

        maxSize = max(bbox[2], bbox[3]) + 120

        if maxSize < imgz[0]:
            maxSize = imgz[0]
        cx, cy = np.array([bbox[0] + bbox[2] / 2,bbox[1] + bbox[3] / 2]).astype(int)
        if (cy + int(maxSize/2))> h:
            cy = h - int(maxSize/2)
        elif cy-int(maxSize/2)<0:
            cy = int(maxSize/2)  

        if (cx + int(maxSize/2))> w:
            cx = w - int(maxSize/2)
        elif cx-int(maxSize/2)<0:
            cx = int(maxSize/2)  

        cropImg = image[int(cy-(maxSize/2)):int(cy+(maxSize/2)), int(cx-(maxSize/2)):int(cx+maxSize/2)]

        offset = np.array([(cx-(maxSize/2)), (cy-(maxSize/2)), maxSize, maxSize]).astype(int) # x,y,w,h

        return cv2.resize(cropImg, imgz, interpolation=cv2.INTER_AREA), offset, mask ,error
    except:
        print(error)


class Ensemble:
    """
    A class to handle image segmentation and object detection tasks using an ensemble of predictions.
    
    Attributes:
    model: Trained machine learning model.
    image: Input image for predictions.
    imageSize: Desired image size (default is (640, 640)).
    task: Task type ('segment' for segmentation, 'obb' for oriented bounding box).
    conf: Confidence threshold for predictions (default is 0.5).
    iou: Intersection over Union threshold for Non-Maximum Suppression (NMS) (default is 0.5).
    show: Boolean to indicate whether to display the predictions on the image (default is False).
    predicted: Stores the final predictions.
    """
    def __init__(self,model, image, imageSize = (640,640), task='segment', conf = 0.5, iou = 0.5, aug = ['original'], show = False):
        """
        Initialize the Ensemble object with provided parameters and resize the input image.
        
        Inputs:
        - model: Trained machine learning model for image segmentation or object detection.
        - image: Input image on which predictions are to be made.
        - imageSize (tuple, optional): Desired image size. Default is (640, 640).
        - task (str, optional): Task type ('segment' or 'obb'). Default is 'segment'.
        - conf (float, optional): Confidence threshold for predictions. Default is 0.5.
        - iou (float, optional): IoU threshold for NMS. Default is 0.5.
        - show (bool, optional): Whether to display the predictions on the image. Default is False.
        
        Outputs:
        - Initializes the Ensemble object with the provided parameters and resized image.
        """
        self.model = model
        self.image = cv2.resize(image, imageSize, interpolation=cv2.INTER_AREA)
        self.imageSize = imageSize
        self.task = task
        self.conf = conf
        self.iou = iou
        self.aug = aug
        self.show = show
        self.predicted = []

    @staticmethod
    def orientedBBox(mask):
        """
        Calculate the oriented bounding box for a given mask.
        
        Inputs:
        - mask: Binary mask from which the oriented bounding box is to be calculated.
        
        Outputs:
        - Returns the coordinates of the oriented bounding box if contours are found, otherwise returns None.
        """
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8) , iterations=3) 
        cnts, _ = cv2.findContours(cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=5), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cnts is not None and len(cnts) > 0:
            mCnt = np.array(max(cnts, key=cv2.contourArea))  
            rect = cv2.minAreaRect(mCnt)
            boxPoints = np.int0(cv2.boxPoints(rect))
            # cv2.drawContours(image, [box_points], 0, (0,255,0), 2)
            return boxPoints
        else:
            return None
    
    @staticmethod
    def augment(image, tasks = ['original']):
        """
        Augment the image by flipping it left-right
        
        Inputs:
        - image: Image to be augmented.
        
        Outputs:
        - Returns a list of augmented images (original, left-right flipped).
        """
        images = []
        for task in tasks:
            if task == 'fliplr':
                img = np.fliplr(image) 
            elif task == 'flipud':
                img = np.flipud(image)
            else:
                img = image
            images.append(img)
        
        return images
    
    def deaugment(self, predicted, task ='original', modelType = 'segment'):
        """
        De-augment the predicted results based on the augmentation type.
        
        Inputs:
        - predicted: List of predicted masks (and boxes for 'obb' task).
        - task (str, optional): Augmentation type ('lr' or 'ud'). Default is 'lr'.
        - image (optional): Image to be used for reference. Default is None.
        - modelType (str, optional): Model type ('segment' or 'obb'). Default is 'segment'.
        
        Outputs:
        - Returns the de-augmented predictions.
        """
        if modelType == 'obb':
            boxes = []
            masks = []
            for mi, mask in enumerate(predicted[0]):
                if task == 'fliplr':
                    mask_ = np.fliplr(mask)
                elif task == 'flipud':
                    mask_ = np.flipud(mask)
                else: # original
                    mask_ = mask
                box = self.orientedBBox(mask_)
                boxes.append(box)
                masks.append(mask_)
            predicted[0] = masks
            predicted[1] = boxes
            return predicted
        elif modelType == 'segment':
            masks = []
            for mi, mask in enumerate(predicted[0]):
                if task == 'fliplr':
                    mask_ = np.fliplr(mask)
                elif task == 'flipud':
                    mask_ = np.flipud(mask)
                else: # original
                    mask_ = mask
                masks.append(mask_)
            predicted[0] = masks
            return predicted

    
    @staticmethod
    def classCombine(results):
        """
        Combine the class results with masks, bounding boxes, class IDs, and scores.
        
        Inputs:
        - results: Model results containing masks, class IDs, and bounding boxes.
        
        Outputs:
        - Returns combined class results with masks, bounding boxes, class IDs, and scores.
        """
        masks = []
        classIDs = []
        scores = []
        rois = []
        for i in range(len(results)):
            if results[i].masks is None:
                continue
            mask = results[i].masks.data.cpu().numpy()
            mask = (mask*255).astype(np.uint8)
            mask = np.squeeze(mask)
            classID = int((results.boxes.data[i][5]).cpu().numpy())
            score = (results.boxes.data[i][4]).cpu().numpy()*100
            roi = (results.boxes.data[i][:4]).cpu().numpy()
            masks.append(mask)  
            scores.append(score)  
            if(classID == 0 or classID ==1):
                classIDs.append(0)  
            else:
                classIDs.append(1)  
            rois.append(roi)  
        return np.array([masks, rois, classIDs, scores], dtype=object)
    
    @staticmethod
    def postprocess(predicted, image):
        """
        Post-process the predictions to refine the masks.
        
        Inputs:
        - predicted: Combined predictions with masks, bounding boxes, class IDs, and scores.
        - image: Image on which predictions were made.
        
        Outputs:
        - Returns post-processed predictions with refined masks.
        """
        masks, rois, classIDs, scores = predicted
        h, w = image.shape[:2]
        for mi, m in enumerate(masks):
            binary_image = np.zeros((h, w), dtype=np.uint8)
            # m = (m*255).astype(np.uint8)
            m = np.squeeze(m)
            contours,_ = cv2.findContours(m,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            mCnt = max(contours, key = cv2.contourArea)
            mask = cv2.drawContours(binary_image, [mCnt], -1, 255, cv2.FILLED)
            kernel = np.ones((3, 3), np.uint8) 
            mask = cv2.erode(mask, kernel, iterations=1) 
            mask = cv2.dilate(mask, kernel, iterations=1) 
            masks[mi]= mask
        return np.array([masks, rois, classIDs, scores], dtype=object)
    
    @staticmethod
    def darkenCheck(predicted):
        """
        Filter out predictions with lower confidence scores for certain class IDs.
        
        Inputs:
        - predicted: Predictions with masks, bounding boxes, class IDs, and scores.
        
        Outputs:
        - Returns predictions with lower confidence scores filtered out.
        """
        masks, rois, classIDs, scores = predicted
        scores_ = []
        indexs = []
        for ci, c in enumerate(classIDs):
            if(c==1):
                scores_.append(scores[ci])
                indexs.append(ci)
        scores_ = np.asarray(scores_)
        if(len(scores_)>1):
            ixs = scores_.argsort()[::-1]
            removeIxs = ixs[0]
            indexs = np.delete(indexs, removeIxs)
            masks = np.delete(masks, indexs)
            rois = np.delete(rois, indexs)
            classIDs = np.delete(classIDs, indexs)
            scores = np.delete(scores, indexs)
        return np.array([masks, rois, classIDs, scores], dtype=object)
    
    @staticmethod
    def obb2Numpy(predicted):
        """
        Convert oriented bounding box predictions to numpy arrays.
        
        Inputs:
        - predicted: Predictions in a specific format containing oriented bounding boxes.
        
        Outputs:
        - Returns numpy arrays of oriented bounding boxes, class IDs, and scores.
        """
        obbRois = []
        obbCls = []
        obbScores = []
        if predicted.obb.cls is None:
            return np.array([obbRois, obbCls, obbScores], dtype=object)
        obbRois_ = predicted.obb.xyxyxyxy.data.cpu().numpy().astype(int)
        obbCls_ = predicted.obb.cls.data.cpu().numpy().astype(int)
        obbScores_ = predicted.obb.conf.data.cpu().numpy()
        for i in range(0, len(obbCls_)):
            obbRois.append(obbRois_[i])
            obbCls.append(obbCls_[i])
            obbScores.append(obbScores_[i])
        return np.array([obbRois, obbCls, obbScores], dtype=object)
    
    @staticmethod
    def obbMaskEstimation(image, obbPred, segPred = None):
        """
        Estimate masks for oriented bounding boxes.
        
        Inputs:
        - image: Image on which predictions were made.
        - obbPred: Oriented bounding box predictions.
        - segPred (optional): Segmentation predictions. Default is None.
        
        Outputs:
        - Returns combined predictions with oriented bounding box masks, rois, class IDs, and scores.
        """
        obbRois = []
        obbCls = []
        obbScores = []
        if(len(obbPred)==3):
            obbRois, obbCls, obbScores = obbPred
        else:
            _, obbRois, obbCls, obbScores = obbPred
        obbMasks = []
        if segPred is not None:
            segMasks, segRois, segClassIDs, segScores = segPred
            for ri,r in enumerate(obbRois):
                binaryObbMask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(binaryObbMask, [r], 0, (255), -1)
                binaryObbMask = np.array(binaryObbMask)
                binarySegMask = segMasks[0]
                intersection = np.logical_and(binarySegMask, binaryObbMask)
                intersectionImage = np.uint8(intersection) * 255
                obbMasks.append(np.uint8(intersection))
        else:
            for ri,r in enumerate(obbRois):
                binaryObbMask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(binaryObbMask, [r], 0, (255), -1)
                obbMasks.append(np.uint8(binaryObbMask))
        return np.array([obbMasks, obbRois, obbCls, obbScores], dtype=object)
    
    def run(self):
        """
        Run the ensemble prediction process on the input image.
        
        Inputs:
        - None (uses attributes from the Ensemble object).
        
        Outputs:
        - Returns final predictions and optionally the image with predictions overlaid if show is True.
        """
        imgs = self.augment(self.image, self.aug)
        results = []

        for i, img in enumerate(imgs):
            result = self.model.predict(source = img, stream=False,conf = self.conf, save_txt  = False)[0]
            if self.task == 'segment':
                result = self.classCombine(result) # BBud, SBud => Bud
            elif self.task == 'obb':
                result = self.obb2Numpy(result)
                result = self.obbMaskEstimation(img, result)

            nmsResult = duplicateRemoval(result, self.iou)
            nmsResult = self.deaugment(nmsResult,task =self.aug[i],modelType = self.task)
            results.append(np.array(nmsResult, dtype=object))
        # results =np.concatenate((np.array(results[0], dtype=object),np.array(results[1], dtype=object),np.array(results[2], dtype=object)),axis=1)
        results =np.concatenate(results,axis=1)
        ensResult = duplicateRemoval(results,self.iou)
        ensResult = duplicateRemoval(ensResult,self.iou-0.2)
        postResult = self.postprocess(ensResult, self.image)
        pred = self.darkenCheck(postResult)
        
        self.predicted = pred
        if(self.show):
            if self.task == 'segment': 
                predImg = aug_plot(self.image, pred[0],pred[2], pred[3])
            elif self.task == 'obb':
                predImg = aug_plot(self.image, pred[0],pred[2], pred[3])
            
            return pred, predImg
        else:
            return pred, self.image
        
class Fusion():
    def __init__(self,models, image, imgz = (640,640), tasks=None, conf = 0.5, iou = 0.5, aug = ['original'], isSeparate= True, show = False):
        """
        Initialize the Fusion class.

        Inputs:
        - models: List of models for segmentation, object detection, and classification.
        - image: Input image for processing.
        - imgz: Image size for model input. Default is (640, 640).
        - tasks: List of tasks for each model. Default is None.
        - conf: Confidence threshold for predictions. Default is 0.5.
        - iou: IOU threshold for duplicate removal. Default is 0.5.
        - show: Boolean flag to display results. Default is False.
        """
        self.models = models
        self.image = image
        self.imgz = imgz
        self.tasks = tasks
        self.conf = conf
        self.iou = iou
        self.aug = aug
        self.isSeparate = isSeparate
        self.show = show

    def classifypReprocessing(self, mask, transparent = False):
        """
        Preprocess the image for classification.

        Inputs:
        - mask: Segmentation mask.
        - transparent: Boolean flag for transparency. Default is False.

        Outputs:
        - Returns the preprocessed image.
        """
        img = self.image.copy()
        mask = np.squeeze(mask[0])
        mask_, mCnt,bbox,center = contourInfor(img, mask, dilate=True)
        masked = img
        if transparent:
            masked = cv2.bitwise_and(img, img, mask=mask_)
        cropImage = cropBBox(masked, bbox)
        maxSize = max(bbox[2], bbox[3])
        minSize = min(bbox[2], bbox[3])
        tmpMasks = np.zeros((maxSize,maxSize,3), dtype="uint8")
        offsetX = (maxSize - bbox[2]) // 2
        offsetY = (maxSize - bbox[3]) // 2
        tmpMasks[offsetY:offsetY+bbox[3], offsetX:offsetX+bbox[2]] = cropImage
        return cv2.resize(tmpMasks,(480,480))
    
    @staticmethod
    def classify2Numpy(predicted):
        """
        Convert classification predictions to NumPy arrays.

        Inputs:
        - predicted: Predicted classification results.

        Outputs:
        - Returns blue and green values as NumPy arrays.
        """
        bVal = predicted[0].probs.data[1].cpu().numpy()
        gVal = predicted[0].probs.data[0].cpu().numpy()
        return bVal, gVal
    
    def obbMaskEstimation(self, obbPred, segPred = None):
        """
        Estimate masks for oriented bounding boxes.

        Inputs:
        - obbPred: Oriented bounding box predictions.
        - segPred (optional): Segmentation predictions. Default is None.

        Outputs:
        - Returns combined predictions with oriented bounding box masks, rois, class IDs, and scores.
        """
        obbRois = []
        obbCls = []
        obbScores = []
        if(len(obbPred)==3):
            obbRois, obbCls, obbScores = obbPred
        else:
            _, obbRois, obbCls, obbScores = obbPred
        obbMasks = []
        if segPred is not None:
            segMasks, segRois, segClassIDs, segScores = segPred
            for ri,r in enumerate(obbRois):
                binary_obbMask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.drawContours(binary_obbMask, [r], 0, (255), -1)
                binary_obbMask = np.array(binary_obbMask)
                binary_segMask = segMasks[0]
                intersection = np.logical_and(binary_segMask, binary_obbMask)
                intersection_image = np.uint8(intersection) * 255
                obbMasks.append(np.uint8(intersection))
        else:
            for ri,r in enumerate(obbRois):
                binary_obbMask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.drawContours(binary_obbMask, [r], 0, (255), -1)
                obbMasks.append(np.uint8(binary_obbMask))
        return np.array([obbMasks, obbRois, obbCls, obbScores], dtype=object)
    

    def darkenEstimation(self, segPred1, segPred2):
        """
        Estimate dark regions in the image.

        Inputs:
        - segPred1: First set of segmentation predictions.
        - segPred2: Second set of segmentation predictions.

        Outputs:
        - Returns the dark mask and an error flag.
        """
        segMasks1, segRois1, segClassIDs1, segScores1 = segPred1
        segMasks2, segRois2, segClassIDs2, segScores2 = segPred2
        fullMask = np.uint8(segMasks2[0]) * 255
        fullMask = cv2.erode(fullMask, np.ones((3, 3), np.uint8) , iterations=1)
        budMask = np.zeros(self.image.shape[:2], dtype="uint8")
        for mi,m in enumerate(segMasks1):
            if segClassIDs1[mi] == 1:
                continue
            budMask =  np.logical_or(budMask, m)
        budMask = np.uint8(budMask) * 255
        budMask = cv2.erode(budMask, np.ones((3, 3), np.uint8) , iterations=1) 
        _, budMask = cv2.threshold(budMask, 127, 255, cv2.THRESH_BINARY_INV)
        masked = cv2.bitwise_and(self.image, self.image, mask=fullMask)
        masked_ = cv2.bitwise_and(masked, masked, mask=budMask)
        tmp = cv2.cvtColor(masked_, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,1,255,cv2.THRESH_BINARY)
        masked_[alpha==0] = [255,255,255]
        b,g,r = cv2.split(masked_)
        gray = cv2.GaussianBlur(g, (9, 9), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        reCenImg = np.zeros(self.image.shape[:2], dtype="uint8")
        cv2.circle(reCenImg,minLoc, 60, 255, -1)
        reCenImg = cv2.bitwise_and(alpha, alpha, mask=reCenImg)
        reCnt = findMaxContour(reCenImg)
        if reCnt is None:
            return 0, 1
        else:
            darkImg = np.zeros(self.image.shape[:2], dtype="uint8")
            cv2.drawContours(darkImg,[reCnt], -1, 255, -1)
            return darkImg, 0
    
    def obbSegClassEstimation(self,segPred1, segPred2, obbPred): # darken part alignment (classID =1 )
        """
        Estimate the class of oriented bounding box segments.

        Inputs:
        - segPred1: First set of segmentation predictions.
        - segPred2: Second set of segmentation predictions.
        - obbPred: Oriented bounding box predictions.

        Outputs:
        - Returns the updated segmentation predictions with class estimation.
        """
        segMasks, segRois, segClassIDs, segScores = segPred1
        obbMasks, obbRois, obbCls, obbScores = obbPred
        segMasks_ = []
        segRois_ = []
        segClassIDs_ = []
        segScores_ = []
        targetSegID  = -1
        targetObbID  = -1
        for si, s in enumerate(segClassIDs):
            if int(s) == 1:
                targetSegID  = si
        for oi, o in enumerate(obbCls):
            if int(o) == 1:
                targetObbID = oi
        if targetSegID ==-1 and targetObbID !=-1:
            _,_,bbox,_ = contourInfor(self.image, obbMasks[targetObbID])
            bbox = np.array(bbox, dtype = np.float32)
            for i,_ in enumerate(segClassIDs):
                segMasks_.append(segMasks[i])
                segRois_.append(segRois[i])
                segClassIDs_.append(segClassIDs[i])
                segScores_.append(segScores[i])
            segMasks_.append(obbMasks[targetObbID])
            segRois_.append(bbox)
            segClassIDs_.append(int(obbCls[targetObbID]))
            segScores_.append( obbScores[targetObbID])
            return  np.array([segMasks_, segRois_, segClassIDs_, segScores_], dtype=object)
        elif targetSegID ==-1 and targetObbID ==-1:

            darkMask, error = self.darkenEstimation(segPred1, segPred2)
            if error == 1:
                return  np.array([segMasks, segRois, segClassIDs, segScores], dtype=object)
            else:
                _,_,bbox,_ = contourInfor(self.image, darkMask)
                bbox = np.array(bbox, dtype = np.float32)
                for i,_ in enumerate(segClassIDs):
                    segMasks_.append(segMasks[i])
                    segRois_.append(segRois[i])
                    segClassIDs_.append(segClassIDs[i])
                    segScores_.append(segScores[i])
                segMasks_.append(darkMask)
                segRois_.append(bbox)
                segClassIDs_.append(1)
                segScores_.append(1)
                return  np.array([segMasks_, segRois_, segClassIDs_, segScores_], dtype=object)
        else:
            return  np.array([segMasks, segRois, segClassIDs, segScores], dtype=object)
        
    def obbSegAlignment(self, segPred, obbPred):
        """
        Align oriented bounding box results with segmentation results.

        Inputs:
        - segPred: Segmentation predictions.
        - obbPred: Oriented bounding box predictions.

        Outputs:
        - Returns an array containing obbMasks, obbRois, obbCls, obbScores, and alignment indices.
        """
        segMasks, segRois, segClassIDs, segScores = segPred
        obbMasks, obbRois, obbCls, obbScores = obbPred
        clIndexs = []
        ixs = obbScores.argsort()[::-1]
        for si,s in enumerate(segMasks):
            if int(segClassIDs[si]) == 1:
                continue
            _,mCnt,_,_ = contourInfor(self.image, s)
            segPoly = Polygon(mCnt.reshape(-1,2))
            removeIxs = []
            for ix in ixs:
                if int(obbCls[ix]) == 1:
                    continue
                obbPoly = Polygon(obbRois[ix])
                centroid = Point(obbPoly.centroid)
                isContains = segPoly.contains(centroid)
                if isContains:
                    clIndexs.append([si,ix])
                    removeIxs = ix
                    break
            removeIxs = np.where(np.asarray(ixs) == removeIxs)[0]
            ixs = np.delete(ixs, removeIxs)
        return np.array([obbMasks, obbRois, obbCls, obbScores, clIndexs], dtype=object)
   
    @staticmethod
    def obbRoiRatio(obbPred):
        """
        Calculate the ratio of the sides of the oriented bounding boxes.

        Inputs:
        - obbPred: Oriented bounding box predictions.

        Outputs:
        - Returns an array containing obbMasks, obbRois, obbClassIDs, obbScores, alignment indices, and obbRoiRatios.
        """
        obbMasks, obbRois, obbClassIDs, obbScores, alignID = obbPred
        obbRoiRatios = []
        for ri, r in enumerate(obbRois):
            if obbClassIDs[ri] ==1:
                obbRoiRatios.append(0)
                continue
            dist1 = distance2D(r[0], r[1])
            dist2 = distance2D(r[0], r[3])
            ratio = 0
            if dist1<=dist2:
                ratio = dist1/dist2
            else:
                ratio = dist2/dist1
            obbRoiRatios.append(ratio)
        return np.array([obbMasks, obbRois, obbClassIDs, obbScores,alignID, obbRoiRatios], dtype=object)
    
    def budDirectionSeg(self, segPred):
        """
        Determine the direction of segmentation masks.

        Inputs:
        - segPred: Segmentation predictions.

        Outputs:
        - Returns arrays of points and reCentroids.
        """
        segMasks, segRois, segCls, segScores = segPred
        centroids = []
        reCentroids = []
        mCnts = []
        pts = []
        for mi, m in enumerate(segMasks):
            mCnt,_,centroid = maxContour(m)
            centroids.append(centroid)
            mCnts.append(mCnt)
        reCentroids = centroids
        darkId =np.where(segCls==1)[0][0]
        for ci,c in enumerate(mCnts):
            if(darkId == ci):
                pts.append(centroids[darkId])
                continue
            pt = closestPointContour(c, centroids[darkId])
            pts.append(pt)
            h, w = self.image.shape[:2]
            tmpMask = np.zeros((h, w, 1), dtype=np.uint8)
            dist = Point(pts[ci]).distance(Point(centroids[ci]))
            mask = np.zeros((h, w, 1), dtype=np.uint8)
            cv2.drawContours(mask, [mCnts[ci]], -1, 255, cv2.FILLED)
            if(dist>80):
                tmpMask = cv2.circle(tmpMask, pts[ci], int(dist), (255,255,255), -1)
                imgMask= cv2.bitwise_and(mask, tmpMask)
                _,_,reCentroid = maxContour(imgMask)
                reCentroids.append(reCentroid)
        return pts, reCentroids
    
    def budDir(self, segMask, startP, dirP):
        """
        Calculate the end point for the direction of a bud.

        Inputs:
        - segMask: Segmentation mask.
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
        endP = projections[id].astype(int)
        return endP
    
    @staticmethod
    def budDirectionObb(point, obb):
        """
        Determine the start and direction points from an oriented bounding box.

        Inputs:
        - point: Reference point.
        - obb: Oriented bounding box.

        Outputs:
        - Returns the start and direction points.
        """
        dists = []
        edgePs = []
        for pi in range(0,len(obb)):
            edgeP = []
            if pi < 3:
                edgeP = midPoint2D(obb[pi],obb[pi+1])
            else:
                edgeP = midPoint2D(obb[pi],obb[0])
            edgePs.append(edgeP)    
            dist = Point(point).distance(Point(edgeP))
            dists.append(dist)
        id = dists.index(max(dists))
        startP = np.array(edgePs[id]).astype(int)
        sID = id +2
        if sID >3:
            sID = abs(4-sID)
        dirP = np.array(edgePs[sID]).astype(int)
        return startP, dirP 

    @staticmethod
    def dirPointCheck(segMask, roiRatio,startP, dirP):
        """
        Check the extrapolated direction point.

        Inputs:
        - segMask: Segmentation mask.
        - roiRatio: Ratio of the sides of the bounding box.
        - startP: Starting point.
        - dirP: Direction point.

        Outputs:
        - Returns the extrapolated point and a boolean indicating if it is contained within the polygon.
        """
        dist = Point(startP).distance(Point(dirP))
        dist = dist *0.15
        exP = linearExtrapolation(startP, dirP, dist)
        cnt,_,_ = maxContour(segMask)
        poly = Polygon(np.squeeze(cnt))
        isContain = poly.contains(Point(exP))
        if roiRatio >=0.97:
            isContain = False
        return np.array(exP).astype(int), isContain

    def obbRoisDir(self, segPred, obbSeg):
        """
        Determine the start, direction, and end points for oriented bounding boxes.

        Inputs:
        - segPred: Segmentation predictions.
        - obbSeg: Oriented bounding box predictions with additional information.

        Outputs:
        - Returns arrays of start points, direction points, end points, and dark centroid.
        """
        startPs = []
        dirPs = []
        endPs = []
        isCon_ = True
        segMasks, segRois, segCls, segScores = segPred
        obbMasks, obbRois, obbCls, obbScores, alignID, obbRoiRatios = obbSeg
        darkCentroid = [0,0]
        if not np.where(segCls==1)[0]:
            pass
        else:
            pts, dirPs = self.budDirectionSeg(segPred)
            # Calculate end points for segCls != 1
            for si,s in enumerate(segMasks):
                if(segCls[si]==1):
                    startPs.append(np.array([0,0]))
                    dirPs.append(np.array([0,0]))
                    endPs.append(np.array([0,0]))
                    _,_,darkCentroid = maxContour(s)
                else:
                    
                    endP = self.budDir(segMasks[si], pts[si], dirPs[si])
                    startPs.append(pts[si])
                    dirPs.append(dirPs[si])
                    endPs.append(endP)
            for ai,a in enumerate(alignID):
                _,_,_,cen = contourInfor(self.image, segMasks[a[0]])
                
                startP, dirP = self.budDirectionObb(cen,obbRois[a[1]])

                
                endP = self.budDir(segMasks[a[0]], startP, dirP)
                
                exP, isCon = self.dirPointCheck(segMasks[a[0]],obbRoiRatios[a[1]], startP, dirP)
                
                if not isCon:
                    isCon_ = False
                else:
                    startPs[a[0]]= startP
                    dirPs[a[0]] = dirP
                    endPs[a[0]]= endP
        return np.array([startPs, dirPs, endPs, darkCentroid], dtype=object)
    
    @staticmethod
    def synchronizationIndex(segPred, obbPred):
        """
        Calculate the synchronization index between segmentation and oriented bounding box predictions.

        Inputs:
        - segPred: Segmentation predictions.
        - obbPred: Oriented bounding box predictions.

        Outputs:
        - Returns the synchronization index.
        """
        segMasks, segRois, segCls, segScores = segPred
        obbMasks, obbRois, obbCls, obbScores = obbPred
        segDark =np.array(np.where(segCls==1)).reshape(-1)
        obbDark =np.array(np.where(obbCls==1)).reshape(-1)
        segNum = len(segCls) - len(segDark)
        obbNum = len(obbCls) - len(obbDark)
        syncIndex = 0
        if (segNum - obbNum) <= 0:
            syncIndex = 0.1
        else: 
            syncIndex = -0.5
        return syncIndex

    def run(self):
        """
        Main function to run all tasks and return the results.

        Inputs:
        - None

        Outputs:
        - Returns the final segmentation predictions, fusion image, bud directions, global view, synchronization index, and error flag.
        """
        try:
        
            error = False
            segPred2 = self.models[1].predict(source = self.image, stream=False,conf = self.conf, save_txt = False)[0]
            segPred2 = toNumby(segPred2)
            segPred2 = duplicateRemoval(segPred2, 0.3)

            segEns = Ensemble(self.models[0], self.image, task=self.tasks[0], aug=self.aug, show = self.show)
            
            segPred1, segImg = segEns.run()
        
            
            if self.isSeparate:
                clsImg = cv2.resize(self.image,(480,480))
                clsPred = self.models[4](clsImg,imgsz = 480, save_txt = False)
                gView,_ = self.classify2Numpy(clsPred)
            else:
                gView =1
            
            
            obbEns = Ensemble(self.models[2], self.image, task=self.tasks[2], aug=self.aug,show = self.show)
        
            obbPred, obbImg = obbEns.run()
            
            obbPred = self.obbMaskEstimation(obbPred, segPred = segPred2)
            
            budSegPred = self.obbSegClassEstimation(segPred1, segPred2, obbPred) #obbSeg
            
            obbSeg = self.obbSegAlignment(segPred1, obbPred)
            
            obbSeg_ = self.obbRoiRatio(obbSeg)
            budDirs = self.obbRoisDir(budSegPred, obbSeg_)
            if self.isSeparate:
                syncIndex = self.synchronizationIndex(budSegPred, obbPred)
            else:
                syncIndex = 0
            
            if self.show:
                segFusionImg = aug_plot(self.image, budSegPred[0],budSegPred[2], budSegPred[3],labels=['bud', 'darken', 'darken'])
                obbFusionImg = aug_plot(self.image, obbSeg[0], obbSeg[2], obbSeg[3], obbRois = obbSeg[1],labels = ['bud', 'darken'], obb = True)
                fusionImg = np.concatenate((segFusionImg,obbFusionImg),axis=0)
                return segPred1, segPred2, budSegPred, fusionImg, budDirs, gView, syncIndex, error
            return segPred1, segPred2, budSegPred, self.image, budDirs, gView, syncIndex, error
        except:
            print("Fusion model error")
    
