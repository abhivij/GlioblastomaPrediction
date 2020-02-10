import torch
import numpy as np
import torch.nn as tnn
import torch.nn.functional as F
import torch.utils.data as data

from dataset import Dataset

import sklearn.metrics as metrics

FEATURE_SIZE = 3368
HIDDEN_LAYER_SIZE = 33

LEARNING_RATE = 0.005

EPOCH = 5

class Network(tnn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = tnn.Linear(FEATURE_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = tnn.Linear(HIDDEN_LAYER_SIZE, 1)            

    def forward(self, input):
        result = self.linear1(input)
        # result = F.sigmoid(self.linear2(result))
        result = self.linear2(result)
        return result

def loss_function():
	# return nn.CrossEntropyLoss()
	return nn.BCEWithLogitsLoss()

def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("Using device: " + str(device))

    data = ''

    network = Network().to(device)
    criterion = loss_function()
    optimiser = torch.optim.Adam(net.parameters(), lr = LEARNING_RATE)  # Minimise the loss using the Adam algorithm.

    for epoch in range(EPOCH):
        running_loss = 0

    	for i in len(data):

    		input = data[i][0]
    		label = data[i][1]

    		optimiser.zero_grad()

    		output = network(input)

    		loss = criterion(output, label)
    		loss.backward()

            optimiser.step()

            running_loss += loss.item()
            print("Epoch : %2d, Loss : %.3f" % (epoch+1, running_loss) )

    with torch.no_grad():
    	prediction = []
    	for i in len(data):
     		input = data[i][0]
    		label = data[i][1]
    		
    		output = torch.round(F.sigmoid(network(input)))   		
    		prediction.append(output)

    labels = data[, 1]
   	accuracy = metrics.accuracy_score(prediction, labels)

   	print("Accuracy : %.3f" % accuracy)

if __name__ == "__main__":
	main()