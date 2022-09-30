#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

def extractFeatures(x : str) -> dict:
    """
    从字符串x中提取特征
    @param string x: 
    @return vector: feature vector representation of x.
    """
    ans = {}
    word_list = x.split(" ")
    for word in word_list:
        if word in ans.keys():
            ans[word] = ans[word] + 1
        else:
            ans[word] = 1
    return ans

def get_random() -> float:
    temp = random.randint(0,200)
    temp = temp - 100
    temp = temp / 100
    return temp

def init_weights(trainExamples : list, testExamples : list) -> dict:
    weights = {}
    for tup in trainExamples:
        sentense = tup[0]
        words = sentense.split(" ")
        for word in words:
            if not word in weights.keys():
                weights[word] = 0
    for tup in testExamples:
        sentense = tup[0]
        words = sentense.split(" ")
        for word in words:
            if not word in weights.keys():
                weights[word] = 0
    return weights

def get_grad(sentense : str, weights : dict) -> float:
    vec = extractFeatures(sentense)
    return dotProduct(vec,weights)

def get_new_weights(emotion, sentense : str, weights, eta) -> dict:
    if get_grad(sentense, weights) >= 1:
        return weights
    vec = extractFeatures(sentense)
    return increment(weights, emotion * eta, vec)

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    给定训练数据和测试数据，特征提取器|featureExtractor|、训练轮数|numIters|和学习率|eta|,
    返回学习后的权重weights
    你需要实现随机梯度下降优化权重
    '''
    weights = init_weights(trainExamples, testExamples)
    for i in range(0, numIters):
        # BEGIN_YOUR_CODE
        for tup in trainExamples:
            sentense = tup[0]
            emotion = tup[1]
            weights = get_new_weights(emotion,sentense,weights,eta)            
        # END_YOUR_CODE
        trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        testError= evaluatePredictor(testExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        print("At iteration %d, error rate on training set is  %f, error rate on test set is %f " % \
            (i, trainError, testError)) 
    return weights


