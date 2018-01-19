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
from mpl_toolkits.mplot3d import Axes3D
colors = matplotlib.colors.cnames




class large_4_gauss(test_base):
	def __init__(self):
		fsize = '200'
		file_name = 'Four_gaussian_3D_' + fsize + '.csv'
		orig_label = 'Four_gaussian_3D_' + fsize + '_original_label.csv'
		alt_label = 'Four_gaussian_3D_' + fsize + '_alt_label.csv'
		test_base.__init__(self, file_name, orig_label, alt_label)

		self.experiment_name = 'Four_Gauss_' + fsize

		self.orig_c_num = 2
		self.orig_sigma = 5

		self.q = 3
		self.sigma_ratio = 0.1
		self.lambda_ratio = 0.2
		self.ISM_exit_count = 10

		self.forced_sigma = 5
		self.forced_lambda = 10	#0.01 , 2


	def plot_original_clustering(self):
		db = self.ASC.db

		X = db['data']
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		Uq_a = np.unique(self.orig_label)
		
		group1 = X[self.orig_label == Uq_a[0]]
		group2 = X[self.orig_label == Uq_a[1]]
		
		ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
		ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
		ax.set_xlabel('Feature 1')
		ax.set_ylabel('Feature 2')
		ax.set_zlabel('Feature 3')
		ax.set_title('Original Clustering')
		plt.show()

	def plot_result(self):
		db = self.ASC.db

		X = db['data']
		fig = plt.figure()
		ax = fig.add_subplot(211, projection='3d')
		Uq_a = np.unique(self.orig_label)
		
		group1 = X[self.orig_label == Uq_a[0]]
		group2 = X[self.orig_label == Uq_a[1]]
		
		ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
		ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
		ax.set_xlabel('Feature 1')
		ax.set_ylabel('Feature 2')
		ax.set_zlabel('Feature 3')
		ax.set_title('Original Clustering')
		
		ax = fig.add_subplot(212, projection='3d')
		Uq_b = np.unique(self.alt_allocation)
		group1 = X[self.alt_allocation == Uq_b[0]]
		group2 = X[self.alt_allocation == Uq_b[1]]
		
		ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
		ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
		ax.set_xlabel('Feature 1')
		ax.set_ylabel('Feature 2')
		ax.set_zlabel('Feature 3')
		ax.set_title('Alternative Clustering')
		
		plt.show()

	def perform_default_run_full(self, debug_info=False):
		test_base.perform_default_run(self, debug_info)
		self.plot_result()

		self.db['cf'].test_2nd_order(self.db)
		#self.plot_convergence_results()
		#import pdb; pdb.set_trace()






D = large_4_gauss()


#	Pick only one Action
#-----	Initialization Runs -----------------------------------
#D.gen_rand_init(n_of_inits=10)
#D.save_ISM_init_to_file()

#-----	Individual Runs -----------------------------------
D.perform_default_run_full(False)
#D.run_with_W_0('ISM', 10, False)
#D.random_initializations(10, 'SM')
#import pdb; pdb.set_trace()
#D.ran_single_from_pickle_init('DG', 3)

#-----	Group Runs -----------------------------------
#D.run_all_based_on_W0()
#D.run_all_random()


#-----	Extra Info -----------------------------------
#print D.ASC.db['W_matrix']
#D.check_local_minimum(D.ASC.db['W_matrix'], 1000, 2)
import pdb; pdb.set_trace()
