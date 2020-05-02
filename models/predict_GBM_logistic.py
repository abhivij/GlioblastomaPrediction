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

def execute_logistic(data, labels):
	print('Logistic Regression --------')
	acc_list = []
	auc_list = []

	data = preprocessing.scale(data)

	training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size = 0.2)
	log_reg_model = LogisticRegression(solver = 'liblinear') 
	log_reg_model.fit(training_data, training_labels)

	params = np.append(log_reg_model.intercept_, log_reg_model.coef_)

	evaluate_model(log_reg_model, training_data, training_labels, 'training data')
	acc, auc = evaluate_model(log_reg_model, test_data, test_labels, 'test data')	

	print('Accuracy : ', acc)
	print('AUC : ', auc)
	print(len(params))


def main():
	data, labels = read_data()
	# execute_logistic(data, labels)



	data = sm.add_constant(data)
	model = sm.Logit(labels, data)
	result = model.fit()
	print(result.summary())



if __name__ == '__main__':
	main()