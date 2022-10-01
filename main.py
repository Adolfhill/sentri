#!/usr/bin/python

import random
import collections
import math
import sys
from util import *
from model import *
from draw import *
from log import *

def TestModel(numIters, eta):
    trainExamples = readExamples('data/data_rt.train')
    testExamples = readExamples('data/data_rt.test')
    featureExtractor = extractFeatures
    weights, trainErrors = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=numIters, eta=eta)
    #trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    testError = evaluatePredictor(testExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print ("train error = %s, test error = %s" % (trainErrors[len(trainErrors) - 1], testError))
    return weights, trainErrors, testError

if __name__ == "__main__":
    logger = getLogger("res/res.INFO")
    numIters = 50
    eta = 0.0
    deltaEta = 0.05
    logger.info("numIters : {}".format(numIters))
    logger.info("(eta , testError),")
    for i in range(20):
        eta = eta + deltaEta
        weights, trainErrors, testError = TestModel(numIters, eta)
        drawAndSave(trainErrors, "./pngs/{}-{}-{}.png".format(numIters, eta, testError))
        logger.info("({} , {})".format(eta, testError))