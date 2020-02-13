import numpy as np

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