SVM Implementation
Binary Classification Task

Output 
Varying accuracies usually ranging from 80 to 97.5 % is found.

Methodology - 
There are two csv fies - one consist of all +1(named as positive) data and other all -1(named as negative) data
The column of Time ,Amount and Class(+1 or -1) are removed.Time and Amount didnt contribute to learning hence removed.
The label file is created inside the code only with first 80 +1 and next 80 -1 ,hence an array of (160,1) for train label and similarly first 20 +1 and next 20 -1 test array of (40,1) generated in code only.
The train and test data of 28 classes are randomly selected(100 +1 and 100 -1).80:20 split is done.
In short we have x_train , y_train , x_test and y_test datas seperately.
The data is modified to put in the cvxopt function
Train on the 80 data and reported accuracy on terminal for the 20 data.
The final accuracy is found by ((number of correct predictions/40)*100)%.

 
