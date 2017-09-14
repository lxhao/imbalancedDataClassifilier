#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
from math import *

from math import sin, cos, pi
from gaft.gaft import GAEngine
from gaft.gaft.analysis.console_output import ConsoleOutputAnalysis
from gaft.gaft.analysis.fitness_store import FitnessStoreAnalysis
from gaft.gaft.components import GAIndividual
from gaft.gaft.components import GAPopulation
from gaft.gaft.operators import FlipBitMutation
from gaft.gaft.operators import RouletteWheelSelection
from gaft.gaft.operators import UniformCrossover
from minepy import MINE
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 从文件读取需要跳过的行数和数据的属性数量
import clusters


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
    if not filename.endswith("CM1.arff"):
      continue
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
def crossValidation(model, x, y, scorer):
  scores = cross_val_score(estimator=model,
                           X=x,
                           y=y,
                           cv=10,
                           n_jobs=-1,
                           scoring='accuracy')
  return np.mean(scores)


def modelCompare(x, y):
  # svmModel = svm.SVC(kernel='rbf', C=195.41, gamma=0.0086, class_weight={1: 4})
  # svmModel.fit(x, y)
  # f1Scorer = make_scorer(f1_score)
  # print('f1:', crossValidation(svmModel, x, y, scorer='f1'))
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

  search(x, y)
  # svm
  # pipeSvc = Pipeline([('scl', StandardScaler()),
  #                     ('clf', SVC(random_state=1, class_weight={1: 10}))])
  # rangeC = np.linspace(1, 1000, num=100)
  # rangeGama = np.linspace(0.001, 0.01, num=100)
  # paramGrid = [{
  #   'clf__C': rangeC,
  #   'clf__kernel': ['rbf'],
  #   'clf__gamma': rangeGama
  # }
  # ]
  # greadSearch = GridSearchCV(estimator=pipeSvc,
  #                            param_grid=paramGrid,
  #                            scoring='f1',
  #                            cv=10,
  #                            n_jobs=-1)
  # greadSearch = greadSearch.fit(x, y)
  # print(greadSearch.best_score_)
  # print(greadSearch.best_params_)


# Define fitness function.
# @engine.fitness_register
def fitness(indv, x, y):
  weight, gama = indv.variants
  print(weight, gama)
  svmModel = SVC(class_weight={1: weight}, gamma=gama)
  f1Scorer = make_scorer(f1_score)
  value = crossValidation(svmModel, x, y, f1Scorer);
  print(value)
  return value


def search(x, y):
  # Define population.
  indv_template = GAIndividual(ranges=[(1, 20), (0, 1)],
                               encoding='binary',
                               eps=0.001)
  population = GAPopulation(indv_template=indv_template, size=50).init()

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
  engine.run(ng=100)


def main():
  dataSets = readData("./data/")
  for filename, data in dataSets.items():
    x = data[:, : -1]
    y = data[:, -1]
    x = x.astype(np.float64)
    y = y.astype(np.int32)
    print('数据集：', filename)
    print('特征数量：', len(x[0]))
    print('记录数量:', len(x))
    posiNum, negaNum = clusters.examplesDistri(y)
    print('正例样本和负例样本的比例为%d:%d' % (posiNum, negaNum))
    # pca = PCA(n_components='mle', svd_solver='full')
    pca = PCA(n_components=10)
    newX = pca.fit_transform(x)
    print('主成分分析后特征数量:', len(newX[0]))
    modelCompare(newX, y)


if __name__ == "__main__":
  main()
