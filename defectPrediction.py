#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
from math import *

from minepy import MINE
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# 从文件读取需要跳过的行数和数据的属性数量
def getSkipRowsAndAttrs(filepath):
  skipRows = 0
  attrs = -1
  with open(filepath) as f:
    line = f.readline()
    while not line.startswith("@data"):
      skipRows += 1
      if line.startswith("@attribute"):
        attrs += 1
      line = f.readline()
  skipRows += 1
  return skipRows, attrs


# 从文件夹读取测试数据
def readData(dirPath):
  result = {}
  for filename in glob.glob(dirPath + '*.arff'):
    # if not filename.endswith("ar4.arff"):
    #   continue
    skipRows, attrs = getSkipRowsAndAttrs(filename)
    data = pd.read_csv(filename, header=None, skiprows=skipRows)
    y = pd.Categorical(data[attrs]).codes
    y.shape = len(y), 1
    data = data.values
    x = data[:, : -1]
    result[filename] = np.hstack((x, y))
  return result


# 统计分对的正类，错分的正类，分对的负类，错分的负类
def resultStatistic(y, yHat):
  TN = TP = FP = FN = 0.0
  for index in range(len(y)):
    if yHat[index] == 1:
      if y[index] == 1:
        TP += 1
      else:
        FP += 1
    else:
      if y[index] == 0:
        TN += 1
      else:
        FN += 1
  return TP, FP, TN, FN


# 用G-mean值评价模型
def scorer(estimator, x, y):
  yHat = estimator.predict(x)
  TP, FP, TN, FN = resultStatistic(y, yHat)
  TPrate = 0.0
  if TP + FN > 0:
    TPrate = TP / (TP + FN)
  TNrate = 0.0
  if TN + FP > 0:
    TNrate = TN / (TN + FP)
  return sqrt(TPrate * TNrate)


# 另外一种模型评价方法，目前没有用到
def modelValidation(model, x, y):
  # 把数据分层抽样
  kfold = StratifiedKFold(y=y, n_folds=10, random_state=1)
  totalTP = totalFP = totalTN = totalFN = 0
  for k, (train, test) in enumerate(kfold):
    model.fit(x[train], y[train])
    yHat = model.predict(x[test])
    TP, FP, TN, FN = resultStatistic(y[test], yHat)
    totalTP += TP
    totalFP += FP
    totalTN += TN
    totalFN += FN
  TPrate = 0.0
  if totalTP + totalFN > 0:
    TPrate = totalTP / (totalTP + totalFN)
  TNrate = 0.0
  FPrate = 0.0
  if totalTN + totalFP > 0:
    TNrate = totalTN / (totalTN + totalFP)
    FPrate = totalFP / (totalTN + totalFP)
  bal = 1 - sqrt((0 - TPrate) ** 2 + (1 - FPrate) ** 2) / sqrt(2)
  gmean = sqrt(TPrate * TNrate)
  return bal, gmean


# 十折交叉验证
def crossValidation(model, x, y):
  scores = cross_val_score(estimator=model,
                           X=x,
                           y=y,
                           cv=10,
                           n_jobs=-1,
                           scoring=scorer)
  return np.mean(scores)


def modelCompare(x, y):
  # # 决策树
  # deTreeModel = DecisionTreeClassifier(criterion='entropy')
  # print(u"决策树模型的f1值", crossValidation(deTreeModel, x, y))
  #
  # # AdaBoost
  # baseEstimator = DecisionTreeClassifier(criterion='entropy')
  # adaBoostModel = AdaBoostClassifier(base_estimator=baseEstimator,
  #                                    n_estimators=10, learning_rate=0.1)
  # print("AdaBoost模型的f1值", crossValidation(adaBoostModel, x, y))
  # # 贝叶斯
  # print("高斯贝叶斯模型的f1值", crossValidation(GaussianNB(), x, y))
  # print("多项式分布贝叶斯模型的f1值", crossValidation(MultinomialNB(), x, y))

  # svm
  pipeSvc = Pipeline([('scl', StandardScaler()),
                      ('clf', SVC(random_state=1, class_weight={1: 10}))])
  rangeC = np.linspace(100, 1000, num=100)
  rangeGama = np.linspace(0, 100, num=200)
  paramGrid = [{
    'clf__C': rangeC,
    'clf__kernel': ['rbf'],
    'clf__gamma': rangeGama
  }
  ]
  greadSearch = GridSearchCV(estimator=pipeSvc,
                             param_grid=paramGrid,
                             scoring=scorer,
                             cv=10,
                             n_jobs=-1)
  greadSearch = greadSearch.fit(x, y)
  print(greadSearch.best_score_)


# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
  m = MINE()
  m.compute_score(x, y)
  return (m.mic(), 0.5)


def main():
  dataSets = readData("./data/")
  for filename, data in dataSets.items():
    print(filename)
    x = data[:, : -1]
    y = data[:, -1]
    x = x.astype(np.float64)
    y = y.astype(np.int32)
    # # 选择K个最好的特征，返回选择特征后的数据
    # # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
    # # 参数k为选择的特征个数
    modelCompare(x, y)


if __name__ == "__main__":
  main()
