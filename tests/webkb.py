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




class webkb(test_base):
	def __init__(self):

		file_name = 'webkb_processed/min_words.csv'
		orig_label = 'webkb_processed/webkbRaw_label_univ.csv'
		alt_label = 'webkb_processed/webkbRaw_label_topic.csv'
		test_base.__init__(self, file_name, orig_label, alt_label)
		self.use_predefined_init_clustering = True

		
		self.experiment_name = 'webkb'
		d_matrix = sklearn.metrics.pairwise.pairwise_distances(self.data, Y=None, metric='euclidean')
		self.orig_sigma = np.median(d_matrix)

		self.orig_c_num = 4

		self.q = 4
		self.c_num = 4
		self.sigma_ratio = 1
		#self.lambda_ratio = 1

		self.ISM_exit_count = 10
		self.forced_lambda = 0.057
		self.forced_sigma = self.median_pair_dist


	def perform_default_run(self, debug_info=False):
		self.write_out(':::::::: ISM default RUN  at ' + self.experiment_name + ' ::::::::\n')
		self.set_up_class(debug_info, debug_info)
		self.predefine_orig_clustering()
		self.calc_alt_cluster()
		self.output_alt_info()
	
	def save_ISM_init_to_file(self):
		self.write_out(':::::::: Save ISM W0  at ' + self.experiment_name + ' ::::::::\n')
		self.set_up_class()
		self.predefine_orig_clustering()

		self.ASC.set_values('save_ISM_init_to_file', True)
		self.ASC.set_values('W_opt_technique', 'ISM')

		self.calc_alt_cluster()





D = webkb()

#	Pick only one Action
#-----	Initialization Runs -----------------------------------
#D.gen_rand_init(n_of_inits=10)
#D.save_ISM_init_to_file()

#-----	Individual Runs -----------------------------------
#D.perform_default_run(False)
#D.run_with_W_0('SM', 10)
D.random_initializations(10, 'SM')
#import pdb; pdb.set_trace()

#-----	Group Runs -----------------------------------
#D.run_all_based_on_W0()
#D.run_all_random()

#D.db['cf'].test_2nd_order(D.db)
import pdb; pdb.set_trace()


