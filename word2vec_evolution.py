# -*- coding: utf-8 -*-
import os
os.getcwd()
os.chdir('/Users/xiaopingguo/Desktop/研究所/研究所研究計畫/中研院資訊研究所研習/講義/stanford NLP/Assignment_Code')

import numpy as nu
from function_softmax_gradcheck_sigmoid_normalizeRow import *
from cost_function import *
from word2vec import *

## center word, out word = (1, n)
## stochastic gradient descent (SGD) - batch

'''
word2vecModel = skipgram
tokens = dict({'a':0, 'b':1, 'c':2, 'd':3, 'e':4})
wordVectors = np.random.randn(10,3)                        

dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)

def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
       for i in range(2*C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext   
word2vecCostAndGradient = softmaxCostAndGradient  
C = 2                             
'''                 
                              
def word2vec_sgd(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    
    ## word2vecModel is "under one center word" function
    ## so ~ this function is designed how many center word we want to focus, it have a a advantage of enhancing speed
    ## cost, grad is under this iteration, what is cost and gradient, so next iteration will use "grad" to change wordVectors
    
    ## C : window size 
    ## wordVectors = inputVectors + outputVectors
    
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]

    ## change center word 50 times under fixed window size
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)


        c, gin, gout = word2vecModel(centerword, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize
        grad[:N/2, :] += gin / batchsize 
        grad[N/2:, :] += gout / batchsize 

    return cost, grad

# print(word2vec_sgd(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient))

