# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:14:21 2018

@author: CRR
"""

#利用logist回归训练贷款违约模型

#逻辑回归 自动建模
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

#参数初始化
filename = 'bankloan.xls'
data = pd.read_excel(filename, 'bankloan')
x = data.iloc[:,:8]#8个属性
y = data.iloc[:,8]#第九列，结果

#稳定性选择方法 挑选特征
rlr = RLR(selection_threshold=0.5) #建立随机逻辑回归模型，筛选变量 特征筛选用了默认阈值0.25
rlr.fit(x,y)#训练模型
t = rlr.get_support()#获取特征筛选结果
#t = np.append(t, False);#需要给特征补齐长度，bool类型的array作为index筛选列时，长度需要和原始数据的列的长度一致
print(u'通过随机逻辑回归模型筛选特征结束。')
print(u'有效特征为：%s' % ','.join(x.columns[t]))

x=data[x.columns[t]] #筛选好特征重新进行模型训练
lr=LR()#建立逻辑回归模型
lr.fit(x,y)#用筛选后的特征数据来训练模型
print(u'逻辑回归模型训练结束。')
print(u'模型的平均正确率为：%s' % lr.score(x,y))