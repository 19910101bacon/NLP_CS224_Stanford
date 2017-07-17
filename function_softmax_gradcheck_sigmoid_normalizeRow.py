import numpy as np
import random
def softmax(x):
    x = np.exp(x)
    if len(x.shape) == 1:
        x = x[np.newaxis]
    x /= np.sum(x, axis = 1)
    return x

#a = softmax(np.random.randn(5,5))
#a = softmax(np.random.random(5))
#print(a)


def grandcheck(f, x):
	fx, grad = f(x)
	h = 1e-4
	it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
	while not it.finished : 
		ix = it.multi_index     #only each element can use multi_index
		x[ix] += h
		before,_ = f(x)
		x[ix] -= 2*h
		after,_ = f(x)
		x[ix] += h
		numgrad = (before - after)/(2*h)
		assert abs(numgrad - grad[ix])/max(1, abs(numgrad), abs(grad[ix])) < 1e-5
		it.iternext()
	print ('gradiant has checked')

#  quad = lambda x : (np.sum(x**2), 2*x )
#  x = np.array(123.5)
#  grandcheck(quad, x)

#  grandcheck(quad, np.random.randn(3,))    # 1-D test
#  grandcheck(quad, np.random.randn(4,5))   # 2-D test
		

def sigmoid(x) : 
	g = 1 + np.exp(-x)
	g = 1/g
	return g


def normalizeRow(x):
	sqr = np.square(x)
	su = np.sum(sqr, axis = 1)
	root = np.sqrt(su)
	y = x / root[np.newaxis].T
	return y








