import numpy as np

# 计算统计特征
class Scaler():
    def __init__(self, data):   # 数据的shape可以是(N,C,L), (N,C) Batchsize、channels、length
        self.data = data
        if self.data.ndim == 3:
            # axis=(0, 2)表示计算会沿着维度0(样本N)和维度2(序列长度L)进行
            # 统计量是按 维度 1 (通道 C) 计算的。例如，self.mean 得到的是每个通道在所有样本和所有序列长度上的平均值。
            self.mean = self.data.mean(axis=(0, 2)).reshape(1, -1, 1)
            self.var = self.data.var(axis=(0, 2)).reshape(1, -1, 1)
            self.max = self.data.max(axis=(0, 2)).reshape(1, -1, 1)
            self.min = self.data.min(axis=(0, 2)).reshape(1, -1, 1)
        elif self.data.ndim == 2:   # (N, C)
            self.mean = self.data.mean(axis=0).reshape(1, -1)
            self.var = self.data.var(axis=0).reshape(1, -1)
            self.max = self.data.max(axis=0).reshape(1, -1)
            self.min = self.data.min(axis=0).reshape(1, -1)
        else:
            raise ValueError('data dim error!')
        
    def standard(self):
        # Z-score标准化
        X = (self.data - self.mean) / (self.var + 1e-6)
        return X

    def minmax(self, feature_range=(0, 1)):
        if feature_range == (0, 1):
            X = (self.data - self.min) / ((self.max - self.min) + 1e-6)
        elif feature_range == (-1, 1):
            X = (self.data - self.min) / ((self.max - self.min) + 1e-6) * 2 - 1
        else:
            raise ValueError('feature_range error!')
        return X