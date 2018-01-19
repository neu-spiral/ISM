#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
from hyper_parameters import *
import numpy.matlib
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import sklearn
import time 
from cost_function import *
import matplotlib.pyplot as plt
from Y_2_allocation import *
import matplotlib 
import calc_cost
from test_base import *
colors = matplotlib.colors.cnames




class data_4(test_base):
	def __init__(self):
		test_base.__init__(self, 'data_4.csv')
		self.experiment_name = 'data_4'

		self.q = 1
		self.orig_c_num = 2
		self.orig_sigma = 0.5
		self.sigma_ratio = 0.2
		self.lambda_ratio = 100
		self.ISM_exit_count = 10

		#	Set the sigma and lambda value to these instead of using ratios.
		self.forced_sigma = 1
		self.forced_lambda = 0.04


		#self.alt_label = np.concatenate((np.ones(10), np.zeros(20), np.ones(10)))
		self.alt_label = np.concatenate((np.ones(20), np.zeros(20)))

	def plot_result(self):
		db = self.ASC.db

		X = db['data']
		plt.figure(1)
		
		plt.subplot(311)
		plt.plot(X[:,0], X[:,1], 'bo')
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title('data_4.csv original plot')
		
		#plt.figure(2)
		plt.subplot(312)
		idx = np.unique(self.orig_label)
		for mm in idx:
			subgroup = X[self.orig_label == mm]
			plt.plot(subgroup[:,0], subgroup[:,1], color=colors.keys()[int(mm)] , marker='o', linestyle='None')
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title('Original Clustering')
		
		
		plt.subplot(313)
		idx = np.unique(self.alt_allocation)
		for mm in idx:
			subgroup = X[self.alt_allocation == mm]
			plt.plot(subgroup[:,0], subgroup[:,1], color=colors.keys()[int(mm)] , marker='o', linestyle='None')
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title('Alternative Clustering')
		
		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
		plt.show()

	def perform_default_run_full(self, debug_info=False, print_allocation=False):
		test_base.perform_default_run(self, debug_info)
		self.plot_result()
		#if print_allocation: self.alt_allocation
		#self.plot_convergence_results()

D = data_4()


#	Pick only one Action
#-----	Initialization Runs -----------------------------------
#D.gen_rand_init(n_of_inits=10)
#D.save_ISM_init_to_file()
#D.find_sigma_lambda()

#-----	Individual Runs -----------------------------------
D.perform_default_run_full(True)
#D.run_with_W_0('ISM', 10, False)
#D.random_initializations(10, 'SM')
#D.ran_single_from_pickle_init('ISM', 1, False)
#D.plot_result()


#-----	Group Runs -----------------------------------
#D.run_all_based_on_W0()
#D.run_all_random()

#D.db['cf'].test_2nd_order(D.db)
import pdb; pdb.set_trace()
