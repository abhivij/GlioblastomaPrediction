"""
	contains models for GBM Vs Healthy prediction using non Neural Network models - namely logistic regression and SVM
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from helper import calculate_aggregate_metric, write_metrics, compute_param_sum, write_model_params

PATH = "../preprocessing/data/output/normalized_GBM_data.csv"

METRIC_COMPUTATION_ITER = 30

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

	return data, labels	

def evaluate_model(model, data, labels, data_name, svm = False, print_details = False):
	outputs = model.predict(data)
	accuracy = accuracy_score(labels, outputs)

	if svm:
		decision_function = model.decision_function(data)
		auc = roc_auc_score(labels, decision_function)
	else:
		predict_proba = np.array([val[1] for val in model.predict_proba(data)])
		auc = roc_auc_score(labels, predict_proba)

	if print_details:
		print('Evaluating on', data_name)
		print("Accuracy : %.3f AUC : %.3f" % (accuracy, auc))

	return accuracy, auc

def main():
	data, labels = read_data()
	data = preprocessing.scale(data)

	print('Logistic Regression --------')
	acc_list = []
	auc_list = []

	coeff_sum = None
	intercept_sum = None
	for _ in range(METRIC_COMPUTATION_ITER):
		training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size = 0.2)
		log_reg_model = LogisticRegression(solver = 'liblinear') 
		log_reg_model.fit(training_data, training_labels)

		coeff_sum, intercept_sum = compute_param_sum(log_reg_model.coef_[0], log_reg_model.intercept_, coeff_sum, intercept_sum)

		evaluate_model(log_reg_model, training_data, training_labels, 'training data')
		acc_tmp, auc_tmp = evaluate_model(log_reg_model, test_data, test_labels, 'test data')	
		acc_list.append(acc_tmp)
		auc_list.append(auc_tmp)
	write_metrics(acc_list, auc_list)

	coeff_mean = coeff_sum / METRIC_COMPUTATION_ITER
	intercept_mean = intercept_sum / METRIC_COMPUTATION_ITER
	write_model_params(coeff_mean, intercept_mean, 'logistic')

	print('\nSVM --------')
	acc_list = []
	auc_list = []	
	for _ in range(METRIC_COMPUTATION_ITER):
		training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size = 0.2)		
		svm_model = svm.SVC(gamma = 'scale', kernel='linear')
		svm_model.fit(training_data, training_labels)

		evaluate_model(svm_model, training_data, training_labels, 'training data', svm = True)
		acc_tmp, auc_tmp = evaluate_model(svm_model, test_data, test_labels, 'test data', svm = True)	
		acc_list.append(acc_tmp)
		auc_list.append(auc_tmp)
	write_metrics(acc_list, auc_list)

if __name__ == '__main__':
	main()