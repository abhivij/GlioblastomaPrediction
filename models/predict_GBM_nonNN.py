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

	return data, labels	

def evaluate_model(model, data, labels, data_name, svm = False):
	outputs = model.predict(data)
	accuracy = accuracy_score(labels, outputs)

	if svm:
		decision_function = model.decision_function(data)
		auc = roc_auc_score(labels, decision_function)
	else:
		predict_proba = np.array([val[1] for val in model.predict_proba(data)])
		auc = roc_auc_score(labels, predict_proba)

	print('Evaluating on', data_name)
	print("Accuracy : %.3f AUC : %.3f" % (accuracy, auc))

def main():
	data, labels = read_data()

	data = preprocessing.scale(data)

	training_data, test_data, training_labels, test_labels = train_test_split(data, labels, test_size = 0.2)

	print('Logistic Regression --------')
	log_reg_model = LogisticRegression(solver = 'liblinear') 
	log_reg_model.fit(training_data, training_labels)

	evaluate_model(log_reg_model, training_data, training_labels, 'training data')
	evaluate_model(log_reg_model, test_data, test_labels, 'test data')	

	print('\nSVM --------')
	svm_model = svm.SVC(gamma = 'scale')
	svm_model.fit(training_data, training_labels)

	evaluate_model(svm_model, training_data, training_labels, 'training data', svm = True)
	evaluate_model(svm_model, test_data, test_labels, 'test data', svm = True)	

if __name__ == '__main__':
	main()