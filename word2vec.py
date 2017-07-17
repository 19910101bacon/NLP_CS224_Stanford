# -*- coding: utf-8 -*-
import os
os.getcwd()
os.chdir('/Users/xiaopingguo/Desktop/研究所/研究所研究計畫/中研院資訊研究所研習/講義/stanford NLP/Assignment_Code')

import numpy as nu
from function_softmax_gradcheck_sigmoid_normalizeRow import *
from cost_function import *

## center word , out word  =  1 , 1

def skipgram(currentWord, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
	
	## currentWord : only one word to represent 'center word'
	## contextWords : list of no more than 2*C strings, the context words, next with center word 
	## token is a dict like {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
	cost = 0.0 ; 
	gradIn_under_vc = np.zeros_like(inputVectors)  # the temp matrix for adjusting [vc] matrix
	gradOut_under_vc = np.zeros_like(outputVectors)  # the temp matrix for adjusting [uw] matrix

	for word in contextWords:
		vc = inputVectors[tokens[currentWord]]  # inputVectors[2:] = inputVectors[2] all present the meaning of selecting third row
		target_out = tokens[word]  # target_out is the number between 1~V
		dcost, dgrad_vc, dgrad_out = word2vecCostAndGradient(vc, target_out, outputVectors, dataset)

		cost += dcost
		gradIn_under_vc[tokens[currentWord]][np.newaxis] += dgrad_vc  # collect gradient of vc under this window(contextWords)
		gradOut_under_vc += dgrad_out

	return cost, gradIn_under_vc, gradOut_under_vc


'''
dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)

def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
       for i in range(2*C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext


dummy_vectors = normalizeRow(np.random.randn(10,3))  # input, output Vectors
dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
s = skipgram("c", ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
print(s)
'''




