# -*- coding: utf-8 -*-
# @Time    : 2020/9/1 16:37
# @Author  : AiVision_YaoHui
# @FileName: roi_pooling.py

import numpy
import six

from chainer import function
from chainer.utils import type_check


def _roi_pooling_slice(size, stride, max_size, roi_offset):
    start = int(numpy.floor(size * stride))
    end = int(numpy.ceil((size + 1) * stride))

    start = min(max(start + roi_offset, 0), max_size)
    end = min(max(end + roi_offset, 0), max_size)

    return slice(start, end), end - start

#将其变为函数
class ROIPooling2D(function.Function):
    """RoI pooling over a set of 2d planes."""

    def __init__(self, outh, outw, spatial_scale):
        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, roi_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            roi_type.dtype == numpy.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 5,
        )

    def forward_cpu(self, inputs):
        bottom_data, bottom_rois = inputs
        n_rois, channels, height, width = bottom_data.shape
        top_data = numpy.empty((n_rois, channels, self.outh, self.outw),
                               dtype=numpy.float32)
        self.argmax_data = numpy.empty_like(top_data).astype(numpy.int32)

        for i_roi in range(n_rois):
            idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
            xmin = int(round(xmin * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))
            ymin = int(round(ymin * self.spatial_scale))
            ymax = int(round(ymax * self.spatial_scale))
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
            strideh = 1. * roi_height / self.outh
            stridew = 1. * roi_width / self.outw

            for outh in range(self.outh):
                sliceh, lenh = _roi_pooling_slice(
                    outh, strideh, height, ymin)
                if sliceh.stop <= sliceh.start:
                    continue
                for outw in range(self.outw):
                    slicew, lenw = _roi_pooling_slice(
                        outw, stridew, width, xmin)
                    if slicew.stop <= slicew.start:
                        continue
                    roi_data = bottom_data[int(idx), :, sliceh, slicew] \
                        .reshape(channels, -1)
                    top_data[i_roi, :, outh, outw] = \
                        numpy.max(roi_data, axis=1)

                    # get the max idx respect to feature_maps coordinates
                    max_idx_slice = numpy.unravel_index(
                        numpy.argmax(roi_data, axis=1), (lenh, lenw))
                    max_idx_slice_h = max_idx_slice[0] + sliceh.start
                    max_idx_slice_w = max_idx_slice[1] + slicew.start
                    max_idx_slice = max_idx_slice_h * width + max_idx_slice_w
                    self.argmax_data[i_roi, :, outh, outw] = max_idx_slice
        return top_data,


    def backward_cpu(self, inputs, gy):
        bottom_data, bottom_rois = inputs
        n_rois, channels, height, width = bottom_data.shape
        bottom_delta = numpy.zeros_like(bottom_data, dtype=numpy.float32)

        for i_roi in range(n_rois):
            idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
            idx = int(idx)
            xmin = int(round(xmin * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))
            ymin = int(round(ymin * self.spatial_scale))
            ymax = int(round(ymax * self.spatial_scale))
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)

            strideh = float(roi_height) / float(self.outh)
            stridew = float(roi_width) / float(self.outw)

            # iterate all the w, h (from feature map) that fall into this ROIs
            for w in range(xmin, xmax + 1):
                for h in range(ymin, ymax + 1):
                    phstart = int(numpy.floor(float(h - ymin) / strideh))
                    phend = int(numpy.ceil(float(h - ymin + 1) / strideh))
                    pwstart = int(numpy.floor(float(w - xmin) / stridew))
                    pwend = int(numpy.ceil(float(w - xmin + 1) / stridew))

                    phstart = min(max(phstart, 0), self.outh)
                    phend = min(max(phend, 0), self.outh)
                    pwstart = min(max(pwstart, 0), self.outw)
                    pwend = min(max(pwend, 0), self.outw)

                    for ph in range(phstart, phend):
                        for pw in range(pwstart, pwend):
                            max_idx_tmp = self.argmax_data[i_roi, :, ph, pw]
                            for c in range(channels):
                                if max_idx_tmp[c] == (h * width + w):
                                    bottom_delta[idx, c, h, w] += \
                                        gy[0][i_roi, c, ph, pw]
        return bottom_delta, None





def roi_pooling_2d(x, rois, outh, outw, spatial_scale):

    # 函数式编程
    return ROIPooling2D(outh, outw, spatial_scale)(x, rois)