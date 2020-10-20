from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 
from numpy import genfromtxt

y_pred = test_mlp('test_data.csv')

test_labels = genfromtxt('test_labels.csv', delimiter=',')

test_accuracy = accuracy(test_labels, y_pred)*100
print('Accuracy = ',test_accuracy,'%')
# print(STUDENT_ID)