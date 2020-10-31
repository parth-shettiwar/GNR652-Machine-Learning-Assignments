import pandas as pd
import random
import numpy as np

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

def main():
	
	f = "positive.csv"
	num_lines = sum(1 for l in open(f))
	size = int(num_lines /4.85)
	skip_idx = random.sample(range(1, num_lines), num_lines - size)
	data = pd.read_csv(f, skiprows=skip_idx )
	train_data = data[:80]
	train_data=np.array(train_data)
	test_data = data[80:]
	
	ones = np.ones((80,1))
	f1 = "negative.csv"
	num_lines = sum(1 for l in open(f1))
	size = int(num_lines /2800)
	
	skip_idx = random.sample(range(1, num_lines), num_lines - size)
	data1 = pd.read_csv(f1, skiprows=skip_idx )
	
	train_data1 = data1[:80]
	test_data1 = data1[80:]
	train_data1=np.array(train_data1)
	oneso=-1*np.ones((80,1))
	oo=np.ones((20,1))
	oo1=-1*np.ones((20,1))
	
	x_train=np.concatenate((train_data,train_data1))
	y_train=np.concatenate((ones,oneso))
	x_test=np.concatenate((test_data,test_data1))
	y_test=np.concatenate((oo,oo1))
	


	from cvxopt import matrix as cvxopt_matrix
	from cvxopt import solvers as cvxopt_solvers


	m,n = x_train.shape
	y = y_train.reshape(-1,1) * 1.
	X_dash = y_train * x_train
	H = np.dot(X_dash , X_dash.T) * 1.


	P = cvxopt_matrix(H)
	q = cvxopt_matrix(-np.ones((m, 1)))
	G = cvxopt_matrix(-np.eye(m))
	h = cvxopt_matrix(np.zeros(m))
	A = cvxopt_matrix(y.reshape(1, -1))
	b = cvxopt_matrix(np.zeros(1))


	cvxopt_solvers.options['show_progress'] = False
	cvxopt_solvers.options['abstol'] = 1e-10
	cvxopt_solvers.options['reltol'] = 1e-10
	cvxopt_solvers.options['feastol'] = 1e-10


	sol = cvxopt_solvers.qp(P, q, G, h, A, b)
	alphas = np.array(sol['x'])
		
	
	w = ((y * alphas).T @ x_train).reshape(-1,1)


	S = (alphas > 1e-4).flatten()


	b = y[S] - np.dot(x_train[S], w)


	
	
	pred=np.dot(x_test,w)+b[0]
	
	p=np.sign(pred)
	count=0
	
	for i in range(0,40):
		if y_test[i]==p[i]:
			count=count+1

	print("Accuracy is", count*2.5)
	
if __name__ == '__main__':
    main()
