# reference code : https://github.com/VafaeeLab/UGVC1056_DeepLearning/blob/master/TEP/ModelL.py

import torch
import numpy as np
import torch.nn as tnn
import torch.nn.functional as F
import torch.utils.data as data

from dataset import Dataset

import sklearn.metrics as metrics

from sklearn.metrics import accuracy_score, roc_auc_score

FEATURE_SIZE = 3368
HIDDEN_LAYER_SIZE = 33

LEARNING_RATE = 0.000009

EPOCH = 300

TRAIN_BATCH_SIZE = 5
TEST_BATCH_SIZE = 5
SET_RATIO = 0.2

PATH = "../preprocessing/data/output/normalized_GBM_data.csv"


class Network(tnn.Module):

	def __init__(self):
		super(Network, self).__init__()
		self.fc1 = tnn.Linear(FEATURE_SIZE, HIDDEN_LAYER_SIZE)
		self.fc2 = tnn.Linear(HIDDEN_LAYER_SIZE, 1)            

	def forward(self, input):
		result = self.fc1(input)
		# result = F.sigmoid(self.fc2(result))
		result = self.fc2(result)
		result = result.flatten()
		return result

def loss_function():
	# return tnn.CrossEntropyLoss()
	return tnn.BCEWithLogitsLoss()

def main():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Using device: " + str(device))

	network = Network().to(device)
	criterion = loss_function()
	optimiser = torch.optim.Adam(network.parameters(), lr = LEARNING_RATE)  # Minimise the loss using the Adam algorithm.

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

	print('\nTraining ...')
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

			if i % training_data_batches == training_data_batches - 1:
				print("Epoch : %2d, Loss : %.3f" % (epoch+1, running_loss) )

	print('\nEvaluating the network')
	evaluate_model(network, training_data, 'training data', device)
	evaluate_model(network, test_data, 'test data', device)


def evaluate_model(network, data, data_name, device):
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

	print('Evaluating on', data_name)
	print("Accuracy : %.3f AUC : %.3f" % (accuracy, auc))


if __name__ == "__main__":
	main()