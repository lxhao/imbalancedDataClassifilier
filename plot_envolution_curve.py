#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt

from best_fit import best_fit

xticks = ['CM1', 'PC2', 'JM1', 'MC2', 'MW1', 'PC1', 'KC1', 'KC3', 'PC5', 'PC4',
          'MC1', 'PC3']


def plot(plt, type):
  for dataSetName in xticks:
    steps, variants, gmeans, f1s = list(zip(*best_fit[dataSetName]))
    best_step, best_v, best_gmean, best_f1 = steps[-1], variants[-1][0], gmeans[
      -1], f1s[-1]
    plt.plot(steps, f1s if type == 'F1' else gmeans, label=dataSetName)
    plt.scatter([best_step], [best_f1] if type == 'F1' else [best_gmean],
                facecolor='r')

    # plt.annotate(s='x: {:.2f}\ny:{:.2f}'.format(best_v, best_gmean),
    #              xy=(best_step, best_gmean),
    #              xytext=(best_step - 0.3, best_gmean - 0.3))


def draw():
  # Set the font dictionaries (for plot title and axis titles)
  title_font = {'fontname': 'SimHei', 'size': '16', 'color': 'black',
                'weight': 'normal',
                'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
  axis_font = {'fontname': 'SimHei', 'size': '14'}

  mpl.rcParams['font.sans-serif'] = [u'SimHei']
  mpl.rcParams['axes.unicode_minus'] = False
  plt.figure(figsize=(12, 6), dpi=120)

  ax = plt.subplot(1, 2, 1)
  # Set the tick labels font
  for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('SimHei')
    label.set_fontsize(13)

  # plt.xticks(range(len(xticks)), xticks, rotation=17)

  plt.axhspan = 1
  plt.axvspan = 0.01
  plot(plt, 'F1')
  plt.legend()
  plt.title(u'F1收敛曲线', **title_font)
  plt.xlabel(u'F1', **axis_font)
  plt.ylabel(u'generation', **axis_font)

  ax = plt.subplot(1, 2, 2)
  # Set the tick labels font
  for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('SimHei')
    label.set_fontsize(13)

  # plt.xticks(range(len(xticks)), xticks, rotation=17)
  plt.axhspan = 1
  plt.axvspan = 0.01
  plot(plt, 'G-mean')
  plt.legend()
  plt.title(u'G-mean收敛曲线', **title_font)
  plt.xlabel(u'G-mean', **axis_font)
  plt.ylabel(u'generation', **axis_font)
  plt.show()


draw()
