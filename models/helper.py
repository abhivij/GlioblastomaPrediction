import numpy as np
from csv import writer
from datetime import datetime

ACC_FILENAME = 'model_acc.csv'
AUC_FILENAME = 'model_auc.csv'

MODEL_PARAM_FILENAME = 'model_param.csv'

DECIMALS = 4

def calculate_aggregate_metric(accuracy_list, auc_list, agg_type = 'min'):
	#returns accuracy and auc which correspond to the minimum total over all elements from the 2 input lists

	if agg_type == 'mean':
		return np.mean(accuracy_list), np.mean(auc_list)
	accuracy = None
	auc = None
	if len(accuracy_list) != len(auc_list):
		print('Length mismatch in Accuracy list and AUC list')
	elif len(accuracy_list) == 1:
		accuracy =  accuracy_list[0]
		auc = auc_list[0]
	else:
		metric_sum_tmp = accuracy_list[0] + auc_list[0]
		metric_sum_min = accuracy_list[0] + auc_list[0]
		min_id = 0
		for i in range(1, len(accuracy_list)):
			metric_sum_tmp = accuracy_list[i] + auc_list[i]
			if metric_sum_min > metric_sum_tmp:
				metric_sum_min = metric_sum_tmp
				min_id = i
		accuracy = accuracy_list[min_id]
		auc = auc_list[min_id]

	return accuracy, auc

def write_metrics(acc_list, auc_list, write_to_file = True, show_all = True):
	acc_list = round_list(acc_list)
	auc_list = round_list(auc_list)

	if show_all:
		print('Accuracy')
		print(','.join([str(e) for e in acc_list]))
		print('AUC')
		print(','.join([str(e) for e in auc_list]))

	if write_to_file:
		acc_filename = datetime.now().strftime("%d-%m-%Y_") + ACC_FILENAME
		with open(acc_filename, 'a+') as write_obj:
			csv_writer = writer(write_obj)
			csv_writer.writerow(acc_list)	
		auc_filename = datetime.now().strftime("%d-%m-%Y_") + AUC_FILENAME
		with open(auc_filename, 'a+') as write_obj:
			csv_writer = writer(write_obj)
			csv_writer.writerow(auc_list)	

	accuracy, auc = calculate_aggregate_metric(acc_list, auc_list)
	print("\nMin aggregate metric\nAccuracy : %.3f AUC : %.3f" % (accuracy, auc))		
	accuracy, auc = calculate_aggregate_metric(acc_list, auc_list, agg_type='mean')
	print("Mean aggregate metric\nAccuracy : %.3f AUC : %.3f" % (accuracy, auc))

	return

def round_list(num_list):
	return [round(e, DECIMALS) for e in num_list]


def compute_param_sum(coeff, intercept, coeff_sum, intercept_sum):
	if coeff_sum is None:
		coeff_sum = coeff
	else:
		coeff_sum = coeff_sum + coeff
	
	if intercept_sum is None:
		intercept_sum = intercept
	else:
		intercept_sum = intercept_sum + intercept

	return coeff_sum, intercept_sum


def write_model_params(coeff, intercept, model_name):
	'''
		writes to a file the model params in the form
		intercept,coeff_feature1,coeff_feature2, ...
	'''
	filename = model_name + '_' + MODEL_PARAM_FILENAME
	coeff = round_list(coeff)
	intercept = round_list(intercept)
	params = str(intercept[0]) + ',' + ','.join([str(e) for e in coeff])
	with open(filename, 'w') as write_obj:
		write_obj.write(params)
	return