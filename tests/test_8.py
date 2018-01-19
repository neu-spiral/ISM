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
colors = matplotlib.colors.cnames




class data_4_test:
	def __init__(self):
		#np.set_printoptions(suppress=True)
		self.data = genfromtxt('data_sets/data_4.csv', delimiter=',')
		self.Y_original = genfromtxt('data_sets/data_4_Y_original.csv', delimiter=',')
		self.U_original = genfromtxt('data_sets/data_4_U_original.csv', delimiter=',')

		self.sigma_ratio=0.1
		self.lambda_ratio=0.1

	def set_up_class(self, debug_1=True, debug_2=True ):
		self.ASC = alt_spectral_clust(self.data)

		self.ASC.set_values('run_debug_1', debug_1)
		self.ASC.set_values('run_debug_2', debug_2)
		self.ASC.set_values('Experiment_name','data_4')


	def calc_original_cluster(self):
		db = self.ASC.db
		if True: #	Calculating the original clustering
			self.ASC.set_values('q',1)
			self.ASC.set_values('C_num',2)
			self.ASC.set_values('sigma',0.5)
		
			self.ASC.set_values('kernel_type','Gaussian Kernel')
			self.ASC.run()
			self.original_allocation = db['allocation']
		
		else: #	Predefining the original clustering, the following are the required settings
			ASC.set_values('q',1)
			ASC.set_values('C_num',2)
			ASC.set_values('sigma',1)
			ASC.set_values('kernel_type','Gaussian Kernel')
			ASC.set_values('W_matrix',np.identity(db['d']))
		
			db['Y_matrix'] = Y_original
			db['U_matrix'] = Y_original
			db['prev_clust'] = 1
			db['allocation'] = Y_2_allocation(Y_original)



	def calc_alt_cluster(self):
		db = self.ASC.db
		[lambdaV, sigma] = get_lambda_sigma(db)
		self.median_pair_dist = sigma
		self.hsic_ratio = lambdaV
		
		self.sigma_used = self.sigma_ratio*sigma
		self.lambda_used = self.lambda_ratio*lambdaV

		self.ASC.set_values('q',2)
		self.ASC.set_values('sigma',self.sigma_used)
		self.ASC.set_values('lambda',self.lambda_used)
	
		
		start_time = time.time() 
		self.ASC.run()
		db['run_alternative_time'] = time.time() - start_time



	def output_alt_info(self):
		db = self.ASC.db
		print("--- %s seconds ---" % db['run_alternative_time'])
	
		self.alt_allocation = db['allocation']	
		a_truth = np.concatenate((np.ones(10), np.zeros(20), np.ones(10)))
		against_alternative = np.around(normalized_mutual_info_score(a_truth,self.alt_allocation), 3)
		against_truth = np.round(normalized_mutual_info_score(self.alt_allocation,self.original_allocation),3)
	
		new_cost = db['cf'].Final_calc_cost(db['W_matrix'])
		CQ = db['cf'].cluster_quality(db)
		Altness = db['cf'].alternative_quality(db)

		print 'sigma used : ', self.sigma_used 
		print 'sigma_ratio : ' , self.sigma_ratio
		print 'median of pairwise distance : ' , self.median_pair_dist

		print 'lambda used : ', self.lambda_used
		print 'lambda_ratio : ' , self.lambda_ratio
		print 'HSIC ratio : ' , self.hsic_ratio

		print 'Alt vs Alt NMI : ' , against_alternative
		print 'Alt vs Orig NMI : ' , against_truth

		print 'Cost : ' , new_cost
		print 'CQ : ' , CQ
		print 'Altness : ' , Altness

		print self.alt_allocation


		#K = db['cf'].create_Kernel(db['W_matrix'])
		#W = db['W_matrix']
		#U = db['U_matrix']
		#print U
		#H = db['H_matrix']
		#D = db['D_matrix']
		#Y = db['Y_matrix']
		#import pdb; pdb.set_trace()



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
		idx = np.unique(self.original_allocation)
		for mm in idx:
			subgroup = X[self.original_allocation == mm]
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
	
	
	def plot_convergence_results(self):
		db = self.ASC.db
		
		if not self.ASC.db['run_debug_1']:
			print 'Cannot plot convergence without run_debug_1 = True'
		if not self.ASC.db['run_debug_2']:
			print 'Cannot plot convergence without run_debug_2 = True'

		X = db['data']
		plt.figure(2)
		
		plt.suptitle('data_4.csv',fontsize=24)
		plt.subplot(311)
		inc = 0
		for costs in db['debug_costVal']: 
			xAxis = np.array(range(len(costs))) + inc; 
			inc = np.amax(xAxis)
			plt.plot(xAxis, costs, 'b')
			plt.plot(inc, costs[-1], 'bo', markersize=10)
			plt.title('Cost vs w iteration, each dot is U update')
			plt.xlabel('w iteration')
			plt.ylabel('cost')
			
		plt.subplot(312)
		inc = 0
		for gradient in db['debug_gradient']: 
			xAxis = np.array(range(len(gradient))) + inc; 
			inc = np.amax(xAxis)
			plt.plot(xAxis, gradient, 'b')
			plt.plot(inc, gradient[-1], 'bo', markersize=10)
			plt.title('Gradient vs w iteration, each dot is U update')
			plt.xlabel('w iteration')
			plt.ylabel('gradient')
	
	
		plt.subplot(313)
		inc = 0
		for wchange in db['debug_debug_Wchange']: 
			xAxis = np.array(range(len(wchange))) + inc; 
			inc = np.amax(xAxis)
			plt.plot(xAxis, wchange, 'b')
			plt.plot(inc, wchange[-1], 'bo', markersize=10)
			plt.title('|w_old - w_new|/|w| vs w iteration, each dot is U update')
			plt.xlabel('w iteration')
			plt.ylabel('|w_old - w_new|/|w| ')
	
		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
		plt.subplots_adjust(top=0.85)
		plt.show()
		


	def output_to_txt(self):
		db = self.ASC.db
		cf = db['cf']
		outLine = str(against_truth) + '\t' + str(against_alternative) + '\t' 
		outLine += str(np.round(cf.cluster_quality(db), 4)) + '\t' + str(np.round(cf.Final_calc_cost(db['W_matrix']),3))
		outLine += '\t' + str(np.round(db['run_alternative_time'],3)) + '\n'
	
		fin = open('Small_Gaussian_result_OM.txt','a')
		fin.write(outLine)
		fin.close()
		

	def maintain_average(self, avg_dict):
		db = self.ASC.db
	
		self.alt_allocation = db['allocation']	
		a_truth = np.concatenate((np.ones(10), np.zeros(20), np.ones(10)))
		against_alternative = np.around(normalized_mutual_info_score(a_truth,self.alt_allocation), 3)
		against_truth = np.round(normalized_mutual_info_score(self.alt_allocation,self.original_allocation),3)
	
		new_cost = db['cf'].Final_calc_cost(db['W_matrix'])
		CQ = db['cf'].cluster_quality(db)
		Altness = db['cf'].alternative_quality(db)

		avg_dict['NMI'].append(against_alternative)
		avg_dict['Alt'].append(against_truth)
		avg_dict['CQ'].append(CQ)
		avg_dict['Cost'].append(new_cost)
		avg_dict['Time'].append(db['run_alternative_time'])





	###############################################
	#	List of actions to perform
	def SM_discover_sigma_lambda_ratio(self):
		lowest_cost = None
		
		for sigma_ratio in np.arange(0.1,2,0.1):
			for lambda_ratio in np.arange(0.1,2,0.1):
				D.set_up_class(False, False)
				db = self.ASC.db

				D.calc_original_cluster()

				self.ASC.set_values('W_opt_technique', 'SM')
				self.ASC.set_values('init_W_from_pickle',True)
				self.ASC.set_values('pickle_count',0)		
				self.calc_alt_cluster()
	
				new_cost = db['cf'].Final_calc_cost(db['W_matrix'])

				if lowest_cost == None: 
					lowest_cost = new_cost
					lowest_sigma_ratio = sigma_ratio
					lowest_lambda_ratio = lambda_ratio 
				else:
					if new_cost < lowest_cost:
						lowest_cost = new_cost
						lowest_sigma_ratio = sigma_ratio
						lowest_lambda_ratio = lambda_ratio 

				
		print 'lowest cost : ' , lowest_cost 
		print 'lowest sigma ratio : ', lowest_sigma_ratio
		print 'lowest lambda ratio : ', lowest_lambda_ratio

	def ISM_discover_sigma_lambda_ratio(self):
		lowest_cost = None
		
		for sigma_ratio in np.arange(0.1,2,0.1):
			for lambda_ratio in np.arange(0.1,2,0.1):
				D.set_up_class(False, False)
				db = self.ASC.db

				D.calc_original_cluster()
				D.calc_alt_cluster()
	
				new_cost = db['cf'].Final_calc_cost(db['W_matrix'])

				if lowest_cost == None: 
					lowest_cost = new_cost
					lowest_sigma_ratio = sigma_ratio
					lowest_lambda_ratio = lambda_ratio 
				else:
					if new_cost < lowest_cost:
						lowest_cost = new_cost
						lowest_sigma_ratio = sigma_ratio
						lowest_lambda_ratio = lambda_ratio 

				
		print 'lowest cost : ' , lowest_cost 
		print 'lowest sigma ratio : ', lowest_sigma_ratio
		print 'lowest lambda ratio : ', lowest_lambda_ratio

	def save_ISM_init_to_file(self):
		self.set_up_class(False, False )
		self.calc_original_cluster()

		self.ASC.set_values('save_ISM_init_to_file', True)
		self.ASC.set_values('W_opt_technique', 'ISM')
		self.ASC.set_values('init_W_from_pickle',True)
		self.ASC.set_values('pickle_count',0)		

		self.calc_alt_cluster()

	def perform_default_run(self, debug_info=False):
		print ':::::::: ISM default RUN  ::::::::'
		self.set_up_class(debug_info, debug_info)
		self.calc_original_cluster()
		self.calc_alt_cluster()
		self.output_alt_info()
		self.plot_result()
		#self.plot_convergence_results()
		import pdb; pdb.set_trace()

	def run_with_W_0(self, technique='SM'):
		debug_info = False
		self.set_up_class(debug_info, debug_info)
		self.calc_original_cluster()

		self.ASC.set_values('W_opt_technique', technique)
		self.ASC.set_values('init_W_from_pickle',True)
		self.ASC.set_values('pickle_count',0)		

		self.calc_alt_cluster()
		self.output_alt_info()
		self.plot_result()
		import pdb; pdb.set_trace()


	def random_initializations(self, n_inits, technique):
		print ':::::::::::::   Random Initialization of ' , technique , ' :::::::::::::'
		debug_info = False
		avg_dict = {}
		avg_dict['NMI'] = []
		avg_dict['Alt'] = []
		avg_dict['CQ'] = []
		avg_dict['Cost'] = []
		avg_dict['Time'] = []

		for pinit in range(n_inits):
			self.set_up_class(debug_info, debug_info)
			self.calc_original_cluster()
			
			self.ASC.set_values('W_opt_technique', technique)
			self.ASC.set_values('init_W_from_pickle',True)
			self.ASC.set_values('pickle_count',pinit)
			self.calc_alt_cluster()

			self.maintain_average(avg_dict)
			self.plot_result()


		NMI = np.array(avg_dict['NMI'])
		CQ = np.array(avg_dict['CQ'])
		Alt = np.array(avg_dict['Alt'])
		Cost = np.array(avg_dict['Cost'])
		Time = np.array(avg_dict['Time'])

		#	Display
		print 'NMI : ' , 	np.round(NMI.mean() , 2), ' ± ' , 	np.round(NMI.std() , 3)
		print 'Alt : ' , 	np.round(Alt.mean() , 2) , ' ± ' , 	np.round(Alt.std(), 3)
		print 'CQ : ' , 	np.round(CQ.mean()  , 2), ' ± ' ,  	np.round(CQ.std(), 3)
		print 'Cost : ' , 	np.round(Cost.mean(), 2) , ' ± ' , 	np.round(Cost.std(), 3)
		print 'Time : ' , 	np.round(Time.mean(), 2) , ' ± ' , 	np.round(Time.std(), 3)

		#	Cut and Paste
		print '\tPast into data table'
		print np.round(NMI.mean() , 2), '±' , 	np.round(NMI.std() , 3), '\t' ,
		print np.round(CQ.mean()  , 2), '±' ,  	np.round(CQ.std(), 3), '\t' ,
		print np.round(Alt.mean() , 2) , '±' , 	np.round(Alt.std(), 3), '\t' ,
		print np.round(Cost.mean(), 2) , '±' , 	np.round(Cost.std(), 3), '\t' ,
		print np.round(Time.mean(), 2) , '±' , 	np.round(Time.std(), 3)




	def gen_rand_init(self, n_of_inits):
		self.set_up_class()
		db = self.ASC.db
		init_W = []

		for p in range(n_of_inits):
			W_temp = np.random.randn(db['d'], db['q']) 			# randomize initialization
			[Q,R] = np.linalg.qr(W_temp)
			init_W.append(Q)

		fname = "init_files/init_W_" + db['Experiment_name'] + ".pk"
		pickle.dump( init_W, open( fname, "wb" ) )


D = data_4_test()

#	Pick only one Action
#D.save_ISM_init_to_file()
#D.ISM_discover_sigma_lambda_ratio()
#D.SM_discover_sigma_lambda_ratio()
#D.gen_rand_init(n_of_inits=10)

#----------------------------------------
D.perform_default_run()
#D.random_initializations(10, 'ISM')
#D.random_initializations(10, 'DG')
#D.run_with_W_0('SM')


