import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from helper import calculate_aggregate_metric, write_metrics, compute_param_sum, write_model_params


import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit



PATH = "../preprocessing/data/output/normalized_GBM_data.csv"

def read_data():
	with open(PATH, 'r') as f:
		string = f.readline()

	tags = string.replace('"', '').strip().split(sep=',')
	tags.pop(0)

	data = np.genfromtxt(PATH, delimiter=',')
	data = data[1:]
	data = np.delete(data, 0, 1)
	data = np.swapaxes(data, 0, 1)

	tag_to_label = {'Cancer':1, 'NonCancer':0}

	labels = []
	for tag in tags:
		labels.append(tag_to_label[tag])
	labels = np.array(labels)

	print('data size', data.shape)
	print('labels size', labels.shape)

	return data, labels	


def evaluate_model(model, data, labels, data_name, print_details = False):
	outputs = model.predict(data)
	accuracy = accuracy_score(labels, outputs)

	predict_proba = np.array([val[1] for val in model.predict_proba(data)])
	auc = roc_auc_score(labels, predict_proba)

	if print_details:
		print('Evaluating on', data_name)
		print("Accuracy : %.3f AUC : %.3f" % (accuracy, auc))

	return accuracy, auc


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def get_logistic_prediction(data, weights):
	n = data.shape[0]
	k = weights.shape[0]

	data = np.hstack((np.ones((n, 1)), data))
	weights = np.reshape(weights, (k, 1))

	print(data.shape)
	print(weights.shape)

	z = np.matmul(data, weights)
	return np.ravel(sigmoid(z))


def execute_logistic(data, labels):
	print('Logistic Regression --------')
	acc_list = []
	auc_list = []

	data = preprocessing.scale(data)

	rs = ShuffleSplit(n_splits = 5, test_size = .2, random_state = 0)
	for train_index, test_index in rs.split(data):
		training_data = data[train_index, :]
		training_labels = labels[train_index]
		test_data = data[test_index, :]
		test_labels = labels[test_index]

		print(training_data.shape)
		print(training_labels.shape)
		print(test_data.shape)
		print(test_labels.shape)

	training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size = 0.2)
	log_reg_model = LogisticRegression(solver = 'liblinear', verbose = 1) 
	log_reg_model.fit(training_data, training_labels)

	params = np.append(log_reg_model.intercept_, log_reg_model.coef_)

	evaluate_model(log_reg_model, training_data, training_labels, 'training data')
	acc, auc = evaluate_model(log_reg_model, test_data, test_labels, 'test data')	

	print('\nAccuracy : ', acc)
	print('AUC : ', auc)
	print(len(params))

	print(log_reg_model.get_params())
	print(log_reg_model.classes_)

	print('\n\nTrying to predict with weights obtained....')
	prediction = get_logistic_prediction(test_data, params)
	print('obtained prediction')

	print('prediction from weights')
	print(prediction)
	print('prediction from sklearn predict_proba')
	sklearn_prediction = np.array([val[1] for val in log_reg_model.predict_proba(test_data)])
	print(sklearn_prediction)
	print('prediction from sklearn predict')
	print(log_reg_model.predict(test_data))
	print('actual test labels')
	print(test_labels)

	# check = prediction - sklearn_prediction
	# print(check)

	plt.plot(params)
	plt.ylabel('LogReg Model Weights')
	plt.show()


	# print('Trying SM')
	# training_data = sm.add_constant(training_data)
	# model = sm.Logit(training_labels, training_data)
	# result = model.fit()
	# print(result.summary())


def main():
	data, labels = read_data()
	execute_logistic(data, labels)







if __name__ == '__main__':
	main()