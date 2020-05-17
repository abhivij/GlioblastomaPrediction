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


import argparse


NUM_SPLITS = 100

def read_data(path):
	with open(path, 'r') as f:
		string = f.readline()

	tags = string.replace('"', '').strip().split(sep=',')
	tags.pop(0)

	data = np.genfromtxt(path, delimiter=',')
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


def execute_logistic(data, labels, penalty, significant_weights = None):
	if not penalty:
		penalty = 'l2'
	print('Logistic Regression with', penalty, 'penalty ....')
	acc_list = []
	auc_list = []

	if significant_weights is not None:
		if significant_weights[0] == 0:
			significant_weights = np.delete(significant_weights, 0)
		significant_weights -= 1
		data = data[:, significant_weights]
		print(data.shape)

	data = preprocessing.scale(data)

	num_params = data.shape[1] + 1
	all_iter_params = np.zeros((NUM_SPLITS, num_params))
	print(all_iter_params.shape)


	rs = ShuffleSplit(n_splits = NUM_SPLITS, test_size = .2, random_state = 0)
	split_count = 0
	for train_index, test_index in rs.split(data):
		training_data = data[train_index, :]
		training_labels = labels[train_index]
		test_data = data[test_index, :]
		test_labels = labels[test_index]

		log_reg_model = LogisticRegression(solver = 'liblinear', penalty = penalty) 
		log_reg_model.fit(training_data, training_labels)

		all_iter_params[split_count, :] = np.append(log_reg_model.intercept_, log_reg_model.coef_)

		evaluate_model(log_reg_model, training_data, training_labels, 'training data')
		acc_tmp, auc_tmp = evaluate_model(log_reg_model, test_data, test_labels, 'test data')	
		acc_list.append(acc_tmp)
		auc_list.append(auc_tmp)

		# print('\nAccuracy : ', acc)
		# print('AUC : ', auc)

		# print(log_reg_model.get_params())
		# print(log_reg_model.classes_)

		# print('\n\nTrying to predict with weights obtained....')
		# prediction = get_logistic_prediction(test_data, all_iter_params[split_count, :])
		# print('obtained prediction')

		# print('prediction from weights')
		# print(prediction)
		# print('prediction from sklearn predict_proba')
		# sklearn_prediction = np.array([val[1] for val in log_reg_model.predict_proba(test_data)])
		# print(sklearn_prediction)
		# print('prediction from sklearn predict')
		# print(log_reg_model.predict(test_data))
		# print('actual test labels')
		# print(test_labels)

		split_count += 1

	write_metrics(acc_list, auc_list, write_to_file = False, show_all = False)

	if significant_weights is not None:
		return


	coeff_mean = np.mean(all_iter_params, axis = 0)
	coeff_se = stats.sem(all_iter_params)

	coeff_z = coeff_mean / coeff_se

	coeff_CI_l = np.percentile(all_iter_params, 2.5, axis = 0)
	coeff_CI_u = np.percentile(all_iter_params, 97.5, axis = 0)


	x = np.array([i for i in range(num_params)])

	plt.plot(x, coeff_mean)
	plt.xlabel('Intercept and Features')
	plt.ylabel('LogReg Model Mean Weights')
	plt.show()

	# plt.plot(x, coeff_se)
	# plt.xlabel('Intercept and Features')
	# plt.ylabel('LogReg Model Weights SE')
	# plt.show()

	# plt.plot(x, coeff_z)
	# plt.xlabel('Intercept and Features')
	# plt.ylabel('LogReg Model Weights Z values')
	# plt.show()

	# fig, ax = plt.subplots()
	# ax.plot(x, coeff_CI_u, label = 'Upper Limit')	
	# ax.plot(x, coeff_CI_l, label = 'Lower Limit')
	# legend = ax.legend(loc='upper left')
	# plt.show()

	fig, ax = plt.subplots()
	ax.plot(x, coeff_mean)
	ax.fill_between(x, coeff_CI_l, coeff_CI_u, color='g')
	plt.show()


	coeff_se_method2 = (coeff_CI_u - coeff_CI_l) / (2 * 1.96)
	coeff_z_method2 = coeff_mean / coeff_se_method2

	# plt.plot(x, coeff_se_method2)
	# plt.xlabel('Intercept and Features')
	# plt.ylabel('LogReg Model Weights SE')
	# plt.show()

	plt.scatter(x, coeff_z_method2, s=2)
	plt.xlabel('Intercept and Features')
	plt.ylabel('LogReg Model Weights Z values')
	plt.show()


	print(len(coeff_mean))
	print(len(coeff_CI_u))
	print(len(coeff_CI_l))
	print(len(coeff_z_method2))

	#for now using method2 since, method2 results for Z look better, with shorter z range


	# plt.hist(all_iter_params[:, 0], density = True)
	# plt.xlabel('Intercept values')
	# plt.show()


	# plt.hist(all_iter_params[:, 1], density = True)
	# plt.xlabel('w1 values')
	# plt.show()

	# plt.hist(all_iter_params[:, 10], density = True)
	# plt.xlabel('w10 values')
	# plt.show()
	
	# significant_weights = np.argwhere(np.absolute(coeff_mean) > 0).flatten()


	significant_weights = np.argwhere(np.absolute(coeff_z_method2) > 2).flatten()
	print('Significant Weights : ', len(significant_weights))

	plt.scatter(significant_weights, coeff_z_method2[significant_weights], s = 2)
	plt.xlabel('Intercept and Features Selected')
	plt.ylabel('LogReg Model Weights Z values')
	plt.show()

	print(significant_weights)


	# print('Trying SM')
	# training_data = sm.add_constant(training_data)
	# model = sm.Logit(training_labels, training_data)
	# result = model.fit()
	# print(result.summary())
	return significant_weights

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--file_path", "-fp", type = str, required = True, help = 'File path including the file name of the data file')
	parser.add_argument("--penalty", "-p", type = str, help = 'penalty used in logistic regression regularization (only supports l1 and l2, default is l2)')
	args = parser.parse_args()

	path = args.file_path
	penalty = args.penalty

	print('Executing the model on', path.split('/')[-1])

	data, labels = read_data(path)
	significant_weights = execute_logistic(data, labels, penalty)

	execute_logistic(data, labels, penalty, significant_weights)





if __name__ == '__main__':
	main()