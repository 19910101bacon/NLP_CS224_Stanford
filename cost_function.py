# -*- coding: utf-8 -*-
import os
os.getcwd()
## os.chdir('/Users/xiaopingguo/Desktop/研究所/研究所研究計畫/中研院資訊研究所研習/講義/stanford NLP/Assignment_Code')

import numpy as nu
import copy
from function_softmax_gradcheck_sigmoid_normalizeRow import *

## input 'center word ' vector
##        which 'out_word' should be focused 
##        another vector(matrix) for word  (any word have two vector(matrix))

def softmaxCostAndGradient(vc, target_out, outputVectors, dataset):
	## outputVectors is the matrix of u, which dimension is (V, d) ; V is the word number, d is the feature
	## vc is one of inputVectors row , so vc is the dimension of (1, d) ; outputVectors is (V,d)
	## there do not have any neuron
	

	V = np.shape(outputVectors)[0]  # all number of word
	d = np.shape(outputVectors)[1]  # feature

	y_hat = softmax(np.dot(vc,outputVectors.T))
	#y_hat = y_hat.sum(axis = 0)  # cut dimension down ; y_hat[np.newaxis] is increase dimension ; reference : https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r
	## target_out is the site of target_out_word, i.e. target is the numeric between 1 ~ V
	## target_out number can correspond to target word and get [0,0,....,1,0,0,...0] which indicate that target word
	delta = copy.copy(y_hat)
    
	delta[:,target_out] -= 1
	cost = -np.log(y_hat[:,target_out])

	grad_vc = np.dot(delta, outputVectors)
	grad_out = np.dot(delta.reshape((V,1)), vc.reshape(1,d))
	return cost, grad_vc, grad_out

'''
## d = 30
vc = np.random.randn(200, 30)[5]   #第六個center
target_out = 24
outputVectors = np.random.randn(2000, 30)

dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)

def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
       for i in range(2*C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext

print(softmaxCostAndGradient(vc, target_out, outputVectors, dataset))
'''


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.
    #
    # Note: See test_word2vec below for dataset's initialization.
    #
    # Input/Output Specifications: same as softmaxCostAndGradient
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    gradPred = np.zeros_like(predicted)
    grad = np.zeros_like(outputVectors)

    indices = [target]
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]

    directions = np.array([1] + [-1 for k in range(K)])

    V = np.shape(outputVectors)[0]
    N = np.shape(outputVectors)[1]

    outputWords = outputVectors[indices,:]

    delta = sigmoid(np.dot(outputWords,predicted) * directions)
    deltaMinus = (delta - 1) * directions;
    cost = -np.sum(np.log(delta));

    gradPred = np.dot(deltaMinus.reshape(1,K+1),outputWords).flatten()
    gradMin = np.dot(deltaMinus.reshape(K+1,1),predicted.reshape(1,N))

    for k in range(K+1):
        grad[indices[k]] += gradMin[k,:]




    # cost = -np.log(sigmoid(np.dot(predicted.reshape(1,N), outputVectors[target].reshape(N,1))));
    # grad[target] = (sigmoid(np.dot(predicted.reshape(1,N), outputVectors[target].reshape(N,1))) - 1) * predicted;
    #
    # gradPred = (sigmoid(np.dot(predicted.reshape(1,N), outputVectors[target].reshape(N,1))) - 1) * outputVectors[target];
    # for i in range(0,K):
    #     randomWord = dataset.sampleTokenIdx()
    #     gradPred -= (sigmoid(np.dot(predicted.reshape(1,N), outputVectors[randomWord].reshape(N,1))) - 1) * outputVectors[randomWord];
    #     grad[randomWord] = (sigmoid(np.dot(predicted.reshape(1,N), outputVectors[randomWord].reshape(N,1))) - 1) * predicted;
    #     cost -= (sigmoid(-np.dot(predicted.reshape(1,N), outputVectors[randomWord].reshape(N,1))));

    ### END YOUR CODE
    # gradPred = gradPred.reshape(N)

    # print np.shape(cost)
    # print "gradpred"
    # print np.shape(gradPred)
    # print np.shape(grad)


    return cost, gradPred, grad
