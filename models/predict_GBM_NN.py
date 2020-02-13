# reference code : https://github.com/VafaeeLab/UGVC1056_DeepLearning/blob/master/TEP/ModelL.py

import torch
import numpy as np
import torch.nn as tnn
import torch.nn.functional as F
import torch.utils.data as data

from dataset import Dataset

import sklearn.metrics as metrics

from sklearn.metrics import accuracy_score, roc_auc_score
from helper import calculate_aggregate_metric

FEATURE_SIZE = 3368
HIDDEN_LAYER_SIZE = 33


LAYER2_SIZE = 1000
LAYER3_SIZE = 100
LAYER4_SIZE = 10

LEARNING_RATE = 0.000009

EPOCH = 300

TRAIN_BATCH_SIZE = 5
TEST_BATCH_SIZE = 5
SET_RATIO = 0.2

METRIC_COMPUTATION_ITER = 5

PATH = "../preprocessing/data/output/normalized_GBM_data.csv"


class SimpleFFNetwork(tnn.Module):

	def __init__(self):
		super(SimpleFFNetwork, self).__init__()
		self.fc1 = tnn.Linear(FEATURE_SIZE, HIDDEN_LAYER_SIZE)
		self.fc2 = tnn.Linear(HIDDEN_LAYER_SIZE, 1)            

	def forward(self, input):
		result = self.fc1(input)
		# result = F.sigmoid(self.fc2(result))
		result = self.fc2(result)
		result = result.flatten()
		return result


class FeedForwardNetwork(tnn.Module):

	def __init__(self):
		super(FeedForwardNetwork, self).__init__()
		self.fc1 = tnn.Linear(FEATURE_SIZE, LAYER2_SIZE)
		self.fc2 = tnn.Linear(LAYER2_SIZE, LAYER3_SIZE)
		self.fc3 = tnn.Linear(LAYER3_SIZE, LAYER4_SIZE)
		self.fc4 = tnn.Linear(LAYER4_SIZE, 1)            

	def forward(self, input):
		result = self.fc1(input)
		result = self.fc2(result)
		result = self.fc3(result)
		result = self.fc4(result)
		result = result.flatten()
		return result


def main():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Using device: " + str(device))

	print('Simple Network : Feedforward 3 layered : 3368-33-1')
	network_function = SimpleFFNetwork
	criterion = tnn.BCEWithLogitsLoss()	# tnn.CrossEntropyLoss()
	
	execute_model(network_function, criterion, device)


	print('Feed Forward Network : 3368-1000-100-10-1')
	network_function = FeedForwardNetwork
	criterion = tnn.BCEWithLogitsLoss()	# tnn.CrossEntropyLoss()

	execute_model(network_function, criterion, device)


def execute_model(network_function, criterion, device, print_details=False):

	acc_list = []
	auc_list = []

	for _ in range(METRIC_COMPUTATION_ITER):
		network = network_function().to(device)
		optimiser = torch.optim.Adam(network.parameters(), lr = LEARNING_RATE)

		df = Dataset(PATH)

		end = int(df.__len__())
		indices = list([i for i in range(0, end)])
		set_split = end - round(end * SET_RATIO)
		train_indices = indices[0 : set_split]
		test_indices = indices[set_split : end]

		training_data = data.DataLoader(df, batch_size = TRAIN_BATCH_SIZE, sampler = data.SubsetRandomSampler(train_indices))
		test_data = data.DataLoader(df, batch_size = TEST_BATCH_SIZE, sampler = data.SubsetRandomSampler(test_indices))

		training_data_batches = len(training_data)
		test_data_batches = len(test_data)

		for epoch in range(EPOCH):
			running_loss = 0

			for i, batch in enumerate(training_data):
				inputs, labels = batch

				inputs, labels = inputs.to(device), labels.to(device)

				optimiser.zero_grad()

				outputs = network(inputs)

				loss = criterion(outputs, labels.type_as(outputs))
				loss.backward()

				optimiser.step()

				running_loss += loss.item()

				if print_details:
					if i % training_data_batches == training_data_batches - 1:
						print("Epoch : %2d, Loss : %.3f" % (epoch+1, running_loss) )

		evaluate_model(network, training_data, 'training data', device)
		acc_tmp, auc_tmp = evaluate_model(network, test_data, 'test data', device, print_details=True)
		acc_list.append(acc_tmp)
		auc_list.append(auc_tmp)

	accuracy, auc = calculate_aggregate_metric(acc_list, auc_list)
	print("\nMin aggregate metric\nAccuracy : %.3f AUC : %.3f" % (accuracy, auc))
	accuracy, auc = calculate_aggregate_metric(acc_list, auc_list, agg_type='mean')
	print("Mean aggregate metric\nAccuracy : %.3f AUC : %.3f" % (accuracy, auc))


def evaluate_model(network, data, data_name, device, print_details=False):
	all_labels = np.array([])

	all_prediction_prob = np.array([])

	with torch.no_grad():
		for inputs, labels in data:
			inputs = inputs.to(device)
			prediction = torch.sigmoid(network(inputs)).cpu().numpy()   		

			all_labels = np.concatenate((all_labels, labels))
			all_prediction_prob = np.concatenate((all_prediction_prob, prediction))

	all_predictions = np.round(all_prediction_prob)
	accuracy = accuracy_score(all_labels, all_predictions)
	auc = roc_auc_score(all_labels, all_prediction_prob)

	if print_details:
		print('Evaluating on', data_name)
		print("Accuracy : %.3f AUC : %.3f" % (accuracy, auc))

	return accuracy, auc


if __name__ == "__main__":
	main()