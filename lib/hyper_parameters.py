#!/usr/bin/env python

import sklearn
import numpy as np


def get_lambda_sigma(db):
	W = np.eye(db['d'])
	db['W_matrix'] = W
	U_matrix = db['U_matrix'].copy()

	db['cf'].init_ISM_matrices(W)
	cost = db['cf'].calc_cost_function(W)

	cq = np.absolute(db['cf'].cluster_quality(db))
	alt = np.absolute(db['cf'].alternative_quality(db))
	lambdaV = cq/alt


	#	Use the median of the pairwise distance as sigma
	d_matrix = sklearn.metrics.pairwise.pairwise_distances(db['data'], Y=None, metric='euclidean')
	sigma = np.median(d_matrix)

	db['W_matrix'] = np.zeros((db['d'], db['q']))
	db['U_matrix'] = U_matrix
	return [lambdaV, sigma]

