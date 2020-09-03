# -*- coding: utf-8 -*-
# @Time    : 2020/9/1 15:58
# @Author  : AiVision_YaoHui
# @FileName: generate_anchors.py

import numpy as np
import torch
import numpy as np
import torch.nn as nn

# 生成单个锚点上9个anchor
def generateAnchors(size_base=16, scales=2 ** np.arange(3, 6), ratios=[0.5, 1, 2]):
    def getWHCxCy(anchor):
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        cx = anchor[0] + 0.5 * (w - 1)
        cy = anchor[1] + 0.5 * (h - 1)
        return w, h, cx, cy

    def makeAnchors(ws, hs, cx, cy):
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((cx - 0.5 * (ws - 1),
                             cy - 0.5 * (hs - 1),
                             cx + 0.5 * (ws - 1),
                             cy + 0.5 * (hs - 1)))
        return anchors

    scales = np.array(scales)
    ratios = np.array(ratios)
    anchor_base = np.array([1, 1, size_base, size_base]) - 1
    w, h, cx, cy = getWHCxCy(anchor_base)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = makeAnchors(ws, hs, cx, cy)
    tmp = list()
    for i in range(anchors.shape[0]):
        w, h, cx, cy = getWHCxCy(anchors[i, :])
        ws = w * scales
        hs = h * scales
        tmp.append(makeAnchors(ws, hs, cx, cy))
    anchors = np.vstack(tmp)
    return torch.from_numpy(anchors).float()


# 依据特征图大小、特征图与原图之间的feature_stride和batch_size生成训练时一个batch中的所有anchor
def generateTotalAnchors(batch_size,fg_probs,feature_width,feature_height,feature_stride,anchor_scales,anchor_ratios):
    # get shift
    anchors = generateAnchors(scales=anchor_scales, ratios=anchor_ratios)
    num_anchors = anchors.size(0)
    shift_x = np.arange(0, feature_width) * feature_stride
    shift_y = np.arange(0, feature_height) * feature_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = torch.from_numpy(
        np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose())
    shifts = shifts.contiguous().type_as(fg_probs).float()
    # get anchors
    anchors = anchors.type_as(fg_probs)
    anchors = anchors.view(1, num_anchors, 4) + shifts.view(shifts.size(0), 1, 4)
    anchors = anchors.view(1, num_anchors * shifts.size(0), 4).expand(batch_size,
                                                                           num_anchors * shifts.size(0), 4)
    return anchors
