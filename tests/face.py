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




class face(test_base):
	def __init__(self):
		#file_name = 'facial_85.csv'
		#orig_label = 'facial_true_labels_624x960.csv'
		#alt_label = 'facial_pose_labels_624x960.csv'

		file_name 	= 'face_624_20.csv'
		orig_label 	= 'face_624_20_alt_label.csv'
		alt_label 	= 'face_624_20_original_label.csv'


		test_base.__init__(self, file_name, orig_label, alt_label, True)
		self.experiment_name = 'face'
		self.use_predefined_init_clustering = True

		#d_matrix = sklearn.metrics.pairwise.pairwise_distances(self.data, Y=None, metric='euclidean')
		#sigma = np.median(d_matrix)

		self.q = 17
		self.orig_c_num = 20
		self.orig_sigma = self.median_pair_dist
		self.c_num = 4

		#self.sigma_ratio = 1
		#self.lambda_ratio = 10
		self.ISM_exit_count = 20

		self.forced_sigma = 0.5*self.median_pair_dist		#	0.5 is working
		self.forced_lambda = 1							#	1 is working


D = face()


#	Pick only one Action
#-----	Initialization Runs -----------------------------------
#D.gen_rand_init(n_of_inits=10)
#D.save_ISM_init_to_file()

#-----	Individual Runs -----------------------------------
D.perform_default_run(True)
#D.run_with_W_0('SM', 10, False)
#D.random_initializations(10, 'ISM')
#import pdb; pdb.set_trace()
#D.ran_single_from_pickle_init('DG', 9, False)

#-----	Group Runs -----------------------------------
#D.run_all_based_on_W0()
#D.run_all_random()


#D.db['cf'].test_2nd_order(D.db)
import pdb; pdb.set_trace()
