1��featuremap_receptivefield_with_hook.py�зֱ������������ͼ��С�͸���Ұ��С��hook������ʹ��ʱ��������hook�������ص�ģ���ϼ��ɵõ�ÿ���featuremap_size��receptive_field_size
2��generate_anchors.py�ļ��а������������ֱ��ǣ�def generateAnchors(size_base=16, scales=2 ** np.arange(3, 6), ratios=[0.5, 1, 2])��def generateTotalAnchors(batch_size,fg_probs,feature_width,feature_height,feature_stride,anchor_scales,anchor_ratios)
����generateAnchors���������һ��ê������9��anchor��generateTotalAnchors��������һ��batch�е�����anchor
3��roi_pooling.py�ļ��а���roi_pooling��forward��backward���й���