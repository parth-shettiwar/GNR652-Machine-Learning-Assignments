import os
import csv
from numpy import genfromtxt
import numpy as np

def cost(x_train,y_train,theta): 
	n=len(y_train)
	
	pred=np.dot(x_train,theta)
	
	cost=(1/2*n)*(np.sum(np.square(-pred+y_train))+0.02*np.sum(np.square(theta)))
	return cost    
def grad(x_train,y_train,theta):
	
	cost_past=np.zeros(500000)
	y_train=np.array(y_train)
	
	
	for k in range(500000):
		
		
		n=len(y_train)
		pred=np.dot(x_train,theta)
		
		xt=np.transpose(x_train)
		
		theta[0]=theta[0]-(1/n)*0.0001*(np.dot(xt,(pred-y_train)))[0]

		for i in range(1,17):
			pred=np.dot(x_train,theta)
			theta[i]=theta[i]-(1/n)*0.004*(np.dot(xt,(pred-y_train)))[i]-(0.01/108)*theta[i]
		
		
		cost_past[k]=cost(x_train,y_train,theta)
		
		
		
	return theta

def main():
	
	
	x_train = genfromtxt('x.csv', delimiter=',')
	y_train = genfromtxt('y.csv', delimiter=',')
	y_train.shape=(108,1)

	x_train=np.array(x_train)
	y_train=np.array(y_train)
	ones = np.ones((108,1))
	x_train = np.column_stack((ones,x_train))
	
	theta = np.random.randn(18,1)
	
	theta=grad(x_train,y_train,theta)
	
	x_test = genfromtxt('x_t.csv', delimiter=',')
	y_test = genfromtxt('y_t.csv', delimiter=',')
	oneso = np.ones((27,1))
	x_test = np.column_stack((oneso,x_test))
	
	pred=np.dot(x_test,theta)
	
	y_test.shape=(27,1)
	print("Weight Matrix is")
	print(theta)
	ss=np.sum(np.square(pred-y_test))
	print("Mean Squared Error on last 27 test set is",ss/27)
	
	
if __name__ == '__main__':
    main()
