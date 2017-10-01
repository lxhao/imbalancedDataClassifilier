#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
import random
from math import *
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import clusters
from gaft.gaft import GAEngine
from gaft.gaft.analysis.console_output import ConsoleOutputAnalysis
from gaft.gaft.analysis.fitness_store import FitnessStoreAnalysis
from gaft.gaft.components import GAIndividual
from gaft.gaft.components import GAPopulation
from gaft.gaft.operators import FlipBitMutation
from gaft.gaft.operators import RouletteWheelSelection
from gaft.gaft.operators import UniformCrossover


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


# 获取标记为label的样本
def getExamples(x, y, label):
  xLabeled = []
  for i in range(len(y)):
    if y[i] == label:
      xLabeled.append(x[i])
  return xLabeled


def stratifiedSampling(x, y, scale):
  xSelected = []
  ySelected = []
  posiNum, negaNum = examplesDistri(y)
  idxSelectedPosi = random.sample(range(int(len(y) * posiNum / len(y))),
                                  int(scale * posiNum / len(y)))
  idxSelectedNega = random.sample(range(int(len(y) * negaNum / len(y))),
                                  int(scale * negaNum / len(y)))
  idxSelectedNega.sort()
  idxSelectedPosi.sort()
  posiCount = 0
  negaCount = 0
  posiIdx = 0
  negaIdx = 0
  for i in range(len(y)):
    if y[i] == 1:
      if posiIdx < len(idxSelectedPosi) and idxSelectedPosi[
        posiIdx] == posiCount:
        xSelected.append(x[i])
        ySelected.append(y[i])
        posiIdx += 1
      posiCount += 1
    else:
      if negaIdx < len(idxSelectedNega) and idxSelectedNega[
        negaIdx] == negaCount:
        xSelected.append(x[i])
        ySelected.append(y[i])
        negaIdx += 1
      negaCount += 1
  return xSelected, ySelected


# 从文件夹读取测试数据
def readData(dirPath):
  result = {}
  for filename in glob.glob(dirPath + '*.arff'):
    # if not filename.endswith("ar5.arff"):
    #   continue
    skipRows, attrs = getSkipRowsAndAttrs(filename)
    data = pd.read_csv(filename, header=None, skiprows=skipRows)
    y = pd.Categorical(data[attrs]).codes
    y.shape = len(y), 1
    data = data.values
    x = data[:, : -1]
    # 样本数量大于300时，随机抽取300份
    if len(y) > 300:
      posiNum, negaNum = examplesDistri(y)
      x, y = stratifiedSampling(x, y, 300)
      posiNum, negaNum = examplesDistri(y)
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
def crossValidation(model, x, y, scorer):
  scores = cross_val_score(estimator=model,
                           X=x,
                           y=y,
                           cv=5,
                           n_jobs=-1,
                           scoring=scorer)
  return np.mean(scores)


# 统计样本正类和负类的数量
# 返回正类和负类样本的数量
def examplesDistri(y):
  positiveCounter = 0
  negativeCounter = 0
  for i in y:
    if i == 1:
      positiveCounter += 1
    else:
      negativeCounter += 1
  return positiveCounter, negativeCounter


def GASvm(x, y):
  x = preprocessing.scale(x)
  search(x, y)
  # # svm
  # posiNum, negaNum = examplesDistri(y)
  # pipeSvc = Pipeline([('clf', SVC(class_weight={1: negaNum / posiNum}))])
  # rangeC = np.linspace(1, 100, num=30)
  # rangeGama = np.linspace(0, 1, num=100)
  # paramGrid = [{
  #   'clf__C': rangeC,
  #   'clf__gamma': rangeGama
  # }
  # ]
  # greadSearch = GridSearchCV(estimator=pipeSvc,
  #                            param_grid=paramGrid,
  #                            scoring=scorer,
  #                            cv=5,
  #                            n_jobs=-1)
  # greadSearch = greadSearch.fit(x, y)
  # bestParams = greadSearch.best_params_
  # bestModel = SVC(C=bestParams['clf__C'], gamma=bestParams['clf__gamma'],
  #                 class_weight={1: negaNum / posiNum})
  # f1 = crossValidation(bestModel, x, y, 'f1')
  # return greadSearch.best_score_, greadSearch.best_params_, f1


# Define fitness function.
# @engine.fitness_register
def fitness(indv, x, y):
  # print(indv.variants)
  weight, C, gama = indv.variants
  gama = gama / 10000
  # print(len(x[0]))
  svmModel = SVC(class_weight={1: weight}, gamma=gama, C=C)
  value = crossValidation(svmModel, x, y, scorer=scorer)
  return value


def search(x, y):
  # Define  weight, C, gamma, features
  features = len(x[0])
  indv_template = GAIndividual(
      ranges=[(1, 20), (1, 1000), (1, 10000)],
      encoding='binary',
      eps=1)
  population = GAPopulation(indv_template=indv_template, size=30).init()

  # Create genetic operators.
  selection = RouletteWheelSelection()
  crossover = UniformCrossover(pc=0.8, pe=0.5)
  mutation = FlipBitMutation(pm=0.1)

  # Create genetic algorithm engine.
  # Here we pass all built-in analysis to engine constructor.
  engine = GAEngine(population=population, selection=selection,
                    crossover=crossover, mutation=mutation,
                    analysis=[ConsoleOutputAnalysis, FitnessStoreAnalysis], x=x,
                    y=y)

  engine.fitness_register(fitness)
  engine.run(ng=5)


def getFileName():
  localTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
  return 'result_' + localTime + '.txt'


def decisionTree(x, y):
  # 决策树
  deTreeModel = DecisionTreeClassifier(criterion='entropy')

  f1 = crossValidation(deTreeModel, x, y, 'f1')
  gmean = crossValidation(deTreeModel, x, y, scorer)
  return gmean, f1


# 随机森林
def randomForest(x, y):
  gmean = crossValidation(RandomForestClassifier(n_estimators=10), x, y, scorer)
  f1 = crossValidation(RandomForestClassifier(n_estimators=10), x, y, 'f1')
  return gmean, f1


def kNN(x, y):
  # 测试k近邻模型的准确率
  k_range = range(1, 31)
  bestGmean = 0
  bestF1 = 0
  for i in k_range:
    gmean = crossValidation(KNeighborsClassifier(n_neighbors=i), x, y, scorer)
    f1 = crossValidation(KNeighborsClassifier(n_neighbors=i), x, y, 'f1')
    if gmean > bestGmean:
      bestGmean = gmean
      bestF1 = f1
  return bestGmean, bestF1


def beyes(x, y):
  f1 = crossValidation(GaussianNB(), x, y, 'f1')
  gmean = crossValidation(GaussianNB(), x, y, scorer)
  return gmean, f1


def adaboost(x, y):
  baseEstimator = DecisionTreeClassifier(criterion='entropy')
  adaBoostModel = AdaBoostClassifier(base_estimator=baseEstimator,
                                     n_estimators=10, learning_rate=0.1)
  f1 = crossValidation(adaBoostModel, x, y, 'f1')
  gmean = crossValidation(adaBoostModel, x, y, scorer)
  return gmean, f1


def svm(x, y):
  x = preprocessing.scale(x)
  pipeSvc = Pipeline([('clf', SVC())])
  rangeC = np.linspace(1, 1000, num=30)
  rangeGama = np.linspace(0, 1, num=100)
  paramGrid = [{
    'clf__C': rangeC,
    'clf__gamma': rangeGama
  }
  ]
  greadSearch = GridSearchCV(estimator=pipeSvc,
                             param_grid=paramGrid,
                             scoring=scorer,
                             cv=5,
                             n_jobs=-1)
  greadSearch = greadSearch.fit(x, y)
  bestParams = greadSearch.best_params_
  bestModel = SVC(C=bestParams['clf__C'], gamma=bestParams['clf__gamma'])
  f1 = crossValidation(bestModel, x, y, 'f1')
  return greadSearch.best_score_, f1


def getDataInfo(filename, data):
  x = data[:, : -1]
  y = data[:, -1]
  f = open('dataInfo.txt', 'w')
  print(filename)
  print('Samples:', len(y))
  print('Attributes:', len(x[0]))
  posiNum, negaNum = examplesDistri(y)
  print('Defect Class:', posiNum)
  print('Non-Defect Class:', negaNum)
  print('Defect:%.2f\n' % (posiNum * 100.0 / (negaNum + posiNum)))


def compare(f, x, y):
  gmean, f1 = svm(x, y)
  f.write('svm模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))
  print('svm模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))

  gmean, f1 = adaboost(x, y)
  f.write('adaboost模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))
  print('adaboost模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))

  gmean, f1 = beyes(x, y)
  f.write('贝叶斯模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))
  print('贝叶斯模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))

  gmean, f1 = randomForest(x, y)
  f.write('随机森林模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))
  print('随机森林模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))

  gmean, f1 = kNN(x, y)
  f.write('k近邻模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))
  print('k近邻模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))

  gmean, f1 = decisionTree(x, y)
  f.write('决策树模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))
  print('决策树模型：\ng-mean值为%.2f,f1值%.2f\n' % (gmean, f1))


def main():
  f = open(getFileName(), 'w')
  dataSets = readData("./data/")
  for filename, data in dataSets.items():
    # if not filename.startswith('./data/CM1'):
    #   continue
    # getDataInfo(filename, data)
    x = data[:, : -1]
    y = data[:, -1]
    x = x.astype(np.float64)
    y = y.astype(np.int32)
    f.write('\n数据集%s\n' % (filename))
    print('\n数据集%s\n' % (filename))
    print('样本数量:', len(y))
    print('特征数量:', len(x[0]))
    # compare(f, x, y)
    GASvm(x, y)


if __name__ == "__main__":
  main()
