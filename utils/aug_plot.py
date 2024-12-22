# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics import YOLO
import os
import numpy as np
import glob
import cv2
import time
import random

# COLORS = ((244,  67,  54),
#           (233,  30,  99),
#           (156,  39, 176),
#           (103,  58, 183),
#           ( 63,  81, 181),
#           ( 33, 150, 243),
#           (  3, 169, 244),
#           (  0, 188, 212),
#           (  0, 150, 136),
#           ( 76, 175,  80),
#           (139, 195,  74),
#           (205, 220,  57),
#           (255, 235,  59),
#           (255, 193,   7),
#           (255, 152,   0),
#           (255,  87,  34),
#           (121,  85,  72),
#           (158, 158, 158),
#           ( 96, 125, 139))
COLORS = ((204, 51, 51),
          (204, 89, 51),
          (204, 128, 51),
          (204, 166, 51),
          (204, 204, 51),
          (166, 204, 51),
          (128, 204, 51),
          (89, 204, 51),
          (51, 204, 51),
          (51, 204, 89),
          (51, 204, 128),
          (51, 204, 166),
          (51, 204, 204),
          (51, 166, 204),
          (51, 128, 204),
          (51, 89, 204),
          (51, 51, 204),
          (89, 51, 204),
          (128, 51, 204),
          (166, 51, 204),
          (204, 51, 204),
          (204, 51, 166),
          (204, 51, 128),
          (204, 51, 89),
          (204, 51, 51))

def plot_one_box(x, img, obbRoi = None, color=None, label=None, line_thickness=3, classify = None, obb =False):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[0]+x[2]), int(x[1]+x[3]))
    c1_ = []
    if obb:
        obbRoisY = obbRoi[:,1]
        id=  np.where(obbRoisY == np.min(obbRoisY))
        c1_ = obbRoi[id[0][0]].reshape(-1)
        c1 = c1_[0],c1_[1]-70
        cv2.drawContours(img, [obbRoi], 0, color, line_thickness, cv2.LINE_AA)

    else:
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        if(classify):
            c2_ = (c2[0], c2[1]-(c1[1] - c2[1]))
            classify_label = 'c_Index ' + str(round(float(classify),2))
            cv2.rectangle(img, c1, c2_, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, classify_label, (c1[0], c2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        else:
            if obb:
                cv2.line(img, c1_, c1, color, thickness=1)
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def overlay(image, mask, color, alpha, resize=None):
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined

def aug_plot(image, masks, classes, scores, obbRois = None,labels = None, classify = None, obb = False):
    colors = []
    scores = np.asarray(scores)
    rois = []
    h, w = image.shape[:2]
    colorIDs = random.sample(range(0, len(COLORS)), len(classes))
    for mi, m in enumerate(masks):
        if(not isinstance(m,np.ndarray)):
            continue
        m = (m*255).astype(np.uint8)
        mask = np.asarray(m).reshape(h, w)
        colors.append(COLORS[colorIDs[mi]])
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        mCnt = max(contours, key = cv2.contourArea)
        
        rect  = cv2.boundingRect(mCnt)#x,y,w,h
        rois.append(rect)
        image =  overlay(image, m, colors[mi], 0.35, resize=None)
    for mi, m in enumerate(masks):
        if(not isinstance(m,np.ndarray)):
            continue    
        if labels is None:
            label_score = str(int(classes[mi])) + " " + str(round(scores[mi],2))
        else:
            label_score = labels[int(classes[mi])] + " " + str(round(scores[mi],2))
        if(classify and  classes[mi]==3):
            image = plot_one_box(rois[mi], image, color=colors[mi],label=label_score, line_thickness=2, classify=classify)
        elif obb:
            image = plot_one_box(rois[mi], image,obbRoi=obbRois[mi], color=colors[mi],label=label_score, line_thickness=2,obb =obb)
        else:
            image = plot_one_box(rois[mi], image, color= colors[mi], label=label_score, line_thickness=2)
    return image

