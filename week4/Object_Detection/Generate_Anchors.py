# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 11:20
# @Author  : AiVision_YaoHui
# @FileName: Generate_Anchors.py

import numpy as np  # 提供矩阵运算功能的库
import cv2

import six



def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):

    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base




# 特征图到原始图的偏移步长
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU

    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

if __name__ == '__main__':  #主函数
    import time
    t = time.time()
    base_anchor = generate_anchor_base()  #生成anchor（窗口）



    image_name = 'Kill_Bill.jpg'
    # Read image
    image = cv2.imread(image_name)

    height,width,_ = image.shape
    anchors  = _enumerate_shifted_anchor(base_anchor,16,height//16,width//16)



    for anchor in base_anchor:
        cv2.rectangle(image,(int(anchor[0]+int(width//2)),int(anchor[1]+int(height//2))),(int(anchor[2]+int(width//2)),int(anchor[3]+int(height//2))),(0,0,255),2)

    cv2.imshow("anchors",image)
    cv2.waitKey()