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




class moon(test_base):
	def __init__(self):
		fsize = '400x4'
		file_name = 'moon_' + fsize + '.csv'
		orig_label = 'moon_' + fsize + '_original_label.csv'
		alt_label = 'moon_' + fsize + '_alt_label.csv'
		test_base.__init__(self, file_name, orig_label, alt_label, True)

		self.experiment_name = 'moon_' + fsize

		self.orig_c_num = 2
		self.orig_sigma = 2

		self.q = 3
		#self.sigma_ratio = 	0.1		# 0.1 works better
		#self.sigma_ratio = 0.1			# 0.1 works better
		#self.lambda_ratio = 500
		self.ISM_exit_count = 10

		self.forced_sigma = 0.10			#	0.29
		self.forced_lambda = 1				#	1

	def plot_result(self):
		X = self.ASC.db['data']
		plt.figure(1)

		plt.subplot(211)
		Uq_a = np.unique(self.orig_label)
		group1 = X[self.orig_label == Uq_a[0]]
		group2 = X[self.orig_label == Uq_a[1]]
	
		#plt.plot(group1[:,0], group1[:,1], 'bo')
		#plt.plot(group2[:,0], group2[:,1], 'ro')
		plt.plot(group1[:,2], group1[:,3], 'bo')
		plt.plot(group2[:,2], group2[:,3], 'ro')
	
		#plt.xlabel('Feature 3')
		#plt.ylabel('Feature 4')
		plt.title('Original Clustering')
		
		
		plt.subplot(212)
		Uq_b = np.unique(self.alt_allocation)
		group1 = X[self.alt_allocation == Uq_b[0]]
		group2 = X[self.alt_allocation == Uq_b[1]]
		#plt.plot(group1[:,2], group1[:,3], 'bo')
		#plt.plot(group2[:,2], group2[:,3], 'ro')
		plt.plot(group1[:,0], group1[:,1], 'bo')
		plt.plot(group2[:,0], group2[:,1], 'ro')
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title('Alternative Clustering')
		
		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
		plt.show()



	def perform_default_run_full(self, debug_info=False):
		test_base.perform_default_run(self, debug_info)
		self.plot_result()
		#self.plot_convergence_results()

	def vary_lambda(self):
		technique = 'DG'
		pickleId = 9
		print technique, pickleId

		for l in np.arange(0.1,4, 0.1):
			self.set_up_class()
			self.calc_original_cluster()
			
			self.ASC.set_values('W_opt_technique', technique)
			self.ASC.set_values('init_W_from_pickle',True)
			self.ASC.set_values('pickle_count', pickleId )
			
			
			self.calc_alt_cluster()
			self.forced_lambda = l
			
			against_alternative = np.around(normalized_mutual_info_score(self.alt_label, self.alt_allocation), 3)
			
			print l, ' : ', against_alternative


D = moon()


#	Pick only one Action
#-----	Initialization Runs -----------------------------------
#D.gen_rand_init(n_of_inits=10)
#D.save_ISM_init_to_file()

#-----	Individual Runs -----------------------------------
D.perform_default_run_full(False)
#D.run_with_W_0('DG', 10)
#D.random_initializations(10, 'ISM')
#D.vary_lambda()
#import pdb; pdb.set_trace()

#-----	Group Runs -----------------------------------
#D.run_all_based_on_W0()
#D.run_all_random()

#D.db['cf'].test_2nd_order(D.db)
import pdb; pdb.set_trace()
