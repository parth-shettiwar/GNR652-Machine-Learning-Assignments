import os
import csv
from numpy import genfromtxt
import numpy as np

def ridge_regression(x_train, y_train, lam): 
    X = np.array(x_train)
    ones = np.ones((108,1))
    X = np.column_stack((ones,X))
    y = np.array(y_train)
    Xt = np.transpose(X)
    lambda_identity = lam*np.identity(len(Xt))
    theInverse = np.linalg.inv(np.dot(Xt, X)+lambda_identity)
    w = np.dot(np.dot(theInverse, Xt), y)
    return w

def main():
	


	x_train = genfromtxt('x.csv', delimiter=',')
	y_train = genfromtxt('y.csv', delimiter=',')
	y_train.shape=(108,1)
	
	x=0
	e=0
	s=10000000000000000
	for tt in range(1):
			lam=0.01
			w=ridge_regression(x_train,y_train,lam)
			x_test=genfromtxt('x_t.csv', delimiter=',')
			ones = np.ones((27,1))
			x_test = np.column_stack((ones,x_test))
			y_test=genfromtxt('y_t.csv', delimiter=',')
			y_test.shape=(27,1)
			
			q=(np.dot(x_test,w))
					
			s=np.sum(np.square(y_test-q))
			
			
			print("Mean squared error is",s/27)
			
	
	
	
if __name__ == '__main__':
    main()
