"""
	logistic regression on PCA of GBM data
"""

import sys
import numpy as np
import pandas as pd
from predict_GBM_nonNN import execute_logistic

PATH = "../visualizations/data/gbm_pca_components.csv"

def read_data():
	tag_to_label = {'Cancer':1, 'NonCancer':0}

	df = pd.read_csv(PATH)

	data = np.array(df.iloc[:, 1:])
	labels = np.array([tag_to_label[e.split('.')[0]] for e in df.iloc[:, 0]])

	return data, labels	

def main():
	data, labels = read_data()

	if len(sys.argv) > 1:
		pc_count = int(sys.argv[1])
	else:
		pc_count = 44			#get top 44 PCA components - corresponds to 90% variance

	data = data[:, :pc_count]	
	print(data.shape)

	execute_logistic(data, labels)



if __name__ == '__main__':
	main()