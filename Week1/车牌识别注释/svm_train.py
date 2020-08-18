#coding: utf-8
import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

#训练图片长宽
SZ = SZ = 20 
PROVINCE_START = 1000

provinces = [
"zh_cuan", "川",
"zh_e", "鄂",
"zh_gan", "赣",
"zh_gan1", "甘",
"zh_gui", "贵",
"zh_gui1", "桂",
"zh_hei", "黑",
"zh_hu", "沪",
"zh_ji", "冀",
"zh_jin", "津",
"zh_jing", "京",
"zh_jl", "吉",
"zh_liao", "辽",
"zh_lu", "鲁",
"zh_meng", "蒙",
"zh_min", "闽",
"zh_ning", "宁",
"zh_qing", "靑",
"zh_qiong", "琼",
"zh_shan", "陕",
"zh_su", "苏",
"zh_sx", "晋",
"zh_wan", "皖",
"zh_xiang", "湘",
"zh_xin", "新",
"zh_yu", "豫",
"zh_yu1", "渝",
"zh_yue", "粤",
"zh_yun", "云",
"zh_zang", "藏",
"zh_zhe", "浙"
]

#将输入图片进行扭曲矫正
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


#计算图像的hog特征
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        
        samples.append(hist)
    return np.float32(samples)

#定义模型的加载/保存方法
class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)  
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        
        #初始化核化SVM
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
    # 定义svm模型训练方法
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    # 定义模型推断方法
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()

    def train_svm(self):
        import pdb
        pdb.set_trace()
        #用于识别英文字母和数字的SVM模型
        self.model = SVM(C=1, gamma=0.5)
        #用于识别中文
        #c为惩罚系数，gamma是选择RBF函数作为kernel后，该函数自带的一个参数
        #gamma值越大支持向量越少，gamma值越小，支持向量越多
        self.modelchinese = SVM(C=1, gamma=0.5)
        #如果本地存在该模型就直接加载，否则读取图片和标签训练模型
        if os.path.exists("./train_dat/svm.dat1"):
            self.model.load("./train_dat/svm.dat")
        else:
            chars_train = []
            chars_label = []
            
            for root, dirs, files in os.walk("./train/chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root,filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    import pdb
                    pdb.set_trace()
                    chars_train.append(digit_img)
                    chars_label.append(root_int)
            
            chars_train = list(map(deskew, chars_train))
            #将训练图像和标签按照对应顺序追加到对应列表中
            chars_train = preprocess_hog(chars_train)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train, chars_label)

        if os.path.exists("./train_dat/svmchinese.dat"):
            self.modelchinese.load("./train_dat/svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk("./train/charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = provinces.index(pinyin) + PROVINCE_START + 1
                for filename in files:
                    filepath = os.path.join(root,filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(index)
            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.modelchinese.train(chars_train, chars_label)

        return self.model, self.modelchinese

    #保存模型
    def save_traindata(self):
        if not os.path.exists("./train_dat/svm.dat"):
            self.model.save("./train_dat/svm.dat")
        if not os.path.exists("./train_dat/svmchinese.dat"):
            self.modelchinese.save("./train_dat/svmchinese.dat")
    
