import numpy as np
from numpy import genfromtxt



STUDENT_NAME = 'SHADMAN RAIHAN'
STUDENT_ID = '20858688'

# defined sigmoid function
def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))
# definded the derivative of sigmoid function
def sigmoid_drv(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))
# defined softmax fucntion
def softmax_func(x):
    val = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    return val


def fwd(inp_data, wt_hid_lyr, bias_hid_lyr,wt_out_lyr,bias_out_lyr):
    '''
    calculating the hidden weighted input for each neuron in hidden layer 
    by multiplying input with hidden weights of that neuron
    and adding the hidden bias values
    '''
    net_hidden = np.dot(inp_data, wt_hid_lyr) + bias_hid_lyr
    '''
    calculating the hidden activations using sigmoid funtion
    '''
    act_hidden = sigmoid_func(net_hidden)
    '''
    calculating the ooutput weighted input for each neuron in output layer 
    by multiplying hidden activations with output weights of that neuron
    and adding the output bias values
    '''
    net_output = np.dot(act_hidden, wt_out_lyr) + bias_out_lyr
    '''
    calculating the output activations using softmax funtion
    '''
    act_output = softmax_func(net_output)
    return act_output, act_hidden, net_hidden

def one_hot_enc(x):
  for i in range(0,len(x)):
    x[i,x[i,:].argmax()]=1
  out = (x == 1).astype(float)
  return out

def test_mlp(data_file):
	# Load the test set
	# START
	test_data=genfromtxt(data_file, delimiter=',')
    # END


	# Load your network
	# START
	weight_hidden = np.load('../Saved Weights/weight_hidden.npy')
	bias_hidden = np.load('../Saved Weights/bias_hidden.npy')
	weight_output = np.load('../Saved Weights/weight_output.npy')
	bias_output = np.load('../Saved Weights/bias_output.npy')
	# END
	y_pred, _, _ = fwd(test_data, weight_hidden, bias_hidden, weight_output, bias_output)

	# Predict test set - one-hot encoded
	y_pred = one_hot_enc(y_pred)
	# y_pred = ...
	
	return y_pred


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''