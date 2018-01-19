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




class Dimond_Gaussians(test_base):
	def __init__(self, dim):
		#dim = '_2_2'
		test_base.__init__(self, 'Dimond_gaussians' + dim + '.csv')
		self.experiment_name = 'Dimond_gaussians' + dim

		self.q = 1
		self.orig_c_num = 2
		self.orig_sigma = 3
		self.sigma_ratio = 0.2
		self.lambda_ratio = 100
		self.ISM_exit_count = 10

		#	Set the sigma and lambda value to these instead of using ratios.
		self.forced_sigma = 6
		self.forced_lambda = 100

		self.use_power_method = False


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
		plt.title('original plot')
		
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
		self.write_out(':::::::: ISM default RUN  at ' + self.experiment_name + ' ::::::::\n')

		self.set_up_class(debug_info, debug_info)					
		self.ASC.set_values('init_W_from_pickle',False)                 
		self.ASC.set_values('run_W_only_and_Stop',True)
		self.ASC.set_values('use_power_method',self.use_power_method)

		if self.use_predefined_init_clustering: self.predefine_orig_clustering()
		else: self.calc_original_cluster()                                                                              
                                                                        
		self.calc_alt_cluster()                                        

		#self.plot_result()
		#if print_allocation: self.alt_allocation
		#self.plot_convergence_results()





def SM_Test_sample_V_time():

	for m in range(1, 4):
		dim = '_' + str(np.power(2, m)) + '_2'
		D = Dimond_Gaussians(dim)
		D.experiment_name = 'Dimond_gaussians' + dim
		D.random_initializations(10, 'SM')





def Test_sample_V_time(store_dic, use_power, dimens):
	domain = []
	ranges = []
	Dims = str(dimens)

	for m in range(6, 15):
		dim = '_' + str(np.power(2, m)) + '_' + Dims
		D = Dimond_Gaussians(dim)
		D.use_power_method = use_power
		D.perform_default_run_full(False)
	
		domain.append(m)
		ranges.append(D.db['run_W_only_time'])
	
		print dim + ' : ' , str(D.db['run_W_only_time'])
	
	r = np.array(ranges)
	r = np.log2(r)
	
	print domain
	print ranges
	print r
	
	store_dic['sample_V_time_'+ Dims + '_' + str(use_power)] = {}
	store_dic['sample_V_time_'+ Dims + '_' + str(use_power)]['log_num_samples'] = domain
	store_dic['sample_V_time_'+ Dims + '_' + str(use_power)]['time_in_s'] = ranges
	store_dic['sample_V_time_'+ Dims + '_' + str(use_power)]['log_time'] = r

	return store_dic
	#plt.subplot(111)
	#plt.plot(domain, r, 'b')
	#plt.xlabel('Log2(Number of samples)')
	#plt.ylabel('Log2(Run Time(s))')
	#plt.title('Num of Samples Vs Time in log2/log2 scale')
	#
	#plt.show()
	#import pdb; pdb.set_trace()

def Test_dimension_V_time(store_dic, use_power, N):
	domain = []
	ranges = []

	for m in range(6, 16):
		dim = '_' + str(N) + '_' + str(np.power(2, m))

		D = Dimond_Gaussians(dim)
		D.use_power_method = use_power
		D.perform_default_run_full(False)
	
		domain.append(m)
		ranges.append(D.db['run_W_only_time'])
	
		print dim + ' : ' , str(D.db['run_W_only_time'])
	
	r = np.array(ranges)
	r = np.log2(r)
	
	print domain
	print ranges
	print r
	
	store_dic['Dim_V_time_N'+ str(N) + '_' + str(use_power)] = {}
	store_dic['Dim_V_time_N'+ str(N) + '_' + str(use_power)]['log_num_samples'] = domain
	store_dic['Dim_V_time_N'+ str(N) + '_' + str(use_power)]['time_in_s'] = ranges
	store_dic['Dim_V_time_N'+ str(N) + '_' + str(use_power)]['log_time'] = r

	return store_dic


	#plt.subplot(111)
	#plt.plot(domain, r, 'b')
	#plt.xlabel('Log2(Number of dimensions)')
	#plt.ylabel('Log2(Run Time(s))')
	#plt.title('Num of Dimension Vs Time in log2/log2 scale')
	#
	#plt.show()
	#import pdb; pdb.set_trace()

def save_info(store_dic):
	fname = 'experiment_results/Size_Dim_V_time.pk'
	if os.path.exists(fname):
		new_Dic = pickle.load( open( fname, "rb" ) )
	else:
		new_Dic = {}


	for i, j in store_dic.items():
		new_Dic[i] = j

	pickle.dump( new_Dic, open( fname, "wb" ) )


def read_info():
	fname = 'experiment_results/Size_Dim_V_time.pk'
	if os.path.exists(fname):
		new_Dic = pickle.load( open( fname, "rb" ) )
	else:
		print 'File ' + fname + ' does not exist'
		raise

	X1 = new_Dic['Dim_V_time_True']['log_num_samples']
	Y1 = new_Dic['Dim_V_time_True']['log_time']
	#new_Dic['Dim_V_time_True']['time_in_s']

	X2 = new_Dic['Dim_V_time_False']['log_num_samples']
	Y2 = new_Dic['Dim_V_time_False']['log_time']
#	#new_Dic['Dim_V_time_False']['time_in_s']


	X3 = new_Dic['sample_V_time_True']['log_num_samples']
	Y3 = new_Dic['sample_V_time_True']['log_time']
	#new_Dic['sample_V_time_True']['time_in_s']

	X4 = new_Dic['sample_V_time_False']['log_num_samples']
	Y4 = new_Dic['sample_V_time_False']['log_time']
	#new_Dic['sample_V_time_False']['time_in_s']


	coef = np.polyfit(X1, Y1, 1)
	p = np.poly1d(coef)
	print coef
	Y1_B = p(X1)

	coef = np.polyfit(X2, Y2, 1)
	print coef
	p = np.poly1d(coef)
	Y2_B = p(X2)

	coef = np.polyfit(X3, Y3, 1)
	print coef
	p = np.poly1d(coef)
	Y3_B = p(X3)


	new_X3 = np.power(2, np.array(X3))

	ax = plt.gca()
	#labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = [0]
	for m in range(len(X3)):
		v = '$2^{' + str(X3[m]) + '}$'
		print v
		labels.append(v)

	ax.set_xticklabels(labels)


	plt.subplot(111)
	plt.plot(X3, Y3, 'go')
	plt.plot(X3, Y3_B, 'b')
	plt.xlabel('Number of samples')
	plt.ylabel('Log2(Run Time(s))')
	plt.title('Num of Samples Vs Time in log2/log2 scale')	
	plt.show()



#	plt.subplot(111)
#	line_up, = plt.plot(X3, Y3, 'g', label='Using Power Method')
#	line_down, = plt.plot(X4, Y4, 'b', label='Not using Power Method')
#	plt.legend(handles=[line_up, line_down])
#	#plt.plot(X1, Y1, 'b')
#	plt.xlabel('Log2(Number of samples)')
#	plt.ylabel('Log2(Run Time(s))')
#	plt.title('Num of Samples Vs Time in log2/log2 scale')	
#	plt.show()


	plt.subplot(111)
	line_up, = plt.plot(X1, Y1, 'go', label='Using Power Method')
	line_down, = plt.plot(X2, Y2, 'bo', label='Not using Power Method')

	plt.plot(X1, Y1_B, 'g')
	plt.plot(X2, Y2_B, 'b')

	plt.legend(handles=[line_up, line_down])
	plt.xlabel('Log2(Number of dimensions)')
	plt.ylabel('Log2(Run Time(s))')
	plt.title('Num of Dimensions Vs Time in log2/log2 scale')
	plt.show()


	import pdb; pdb.set_trace()


#for m in range(1,13):
#	dim = '_' + str(np.power(2, m)) + '_2'
#	D = Dimond_Gaussians(dim)
#	D.gen_rand_init(n_of_inits=10)

#SM_Test_sample_V_time()
read_info()

#print 'No Power method sample V time'
#Test_sample_V_time(store_dic, False)
#print 'No Power method dimenssion V time'



#store_dic = {}
#Test_sample_V_time(store_dic, False, 2)
#Test_sample_V_time(store_dic, False, 4)
#Test_sample_V_time(store_dic, False, 8)
#
#Test_sample_V_time(store_dic, True, 2)
#Test_sample_V_time(store_dic, True, 4)
#Test_sample_V_time(store_dic, True, 8)
#save_info(store_dic)



#store_dic = {}
#store_dic = Test_dimension_V_time(store_dic, True, 64)
#store_dic = Test_dimension_V_time(store_dic, False, 64)
#store_dic = Test_dimension_V_time(store_dic, True, 128)
#store_dic = Test_dimension_V_time(store_dic, False, 128)
#store_dic = Test_dimension_V_time(store_dic, True, 256)
#store_dic = Test_dimension_V_time(store_dic, False, 256)
#
#save_info(store_dic)




#print 'Power method sample V time'
#Test_sample_V_time(store_dic, True)
#print 'Power method dimenssion V time'
#Test_dimension_V_time(store_dic, True)
