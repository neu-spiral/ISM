#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
import os
import uuid
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
from sklearn import preprocessing
colors = matplotlib.colors.cnames




class test_base:
	def __init__(self, data_file=None, orig_label=None, alt_label=None, preprocess_data=False):
		if data_file != None : 
			self.data = genfromtxt('data_sets/' + data_file, delimiter=',')
			if preprocess_data: self.data = preprocessing.scale(self.data)
		if orig_label != None : self.orig_label = genfromtxt('data_sets/' + orig_label, delimiter=',')
		if alt_label != None : self.alt_label = genfromtxt('data_sets/' + alt_label, delimiter=',')

		d_matrix = sklearn.metrics.pairwise.pairwise_distances(self.data, Y=None, metric='euclidean')
		self.median_pair_dist = np.median(d_matrix)

		

		self.sigma_ratio = 1
		self.lambda_ratio = 1
		self.q = 2
		self.orig_c_num = 2
		self.orig_sigma = 2
		self.c_num = 2
		self.ISM_exit_count = 10
		self.forced_lambda = None
		self.forced_sigma = None
		self.experiment_name = 'NOT_SET'
		self.use_predefined_init_clustering = False

	def set_up_class(self, debug_1=False, debug_2=False):
		self.ASC = alt_spectral_clust(self.data)

		self.ASC.set_values('ISM_exit_count',self.ISM_exit_count)
		self.ASC.set_values('run_debug_1', debug_1)
		self.ASC.set_values('run_debug_2', debug_2)
		self.ASC.set_values('Experiment_name', self.experiment_name)
		self.ASC.set_values('run_hash', str(uuid.uuid4()))
		self.ASC.set_values('q',self.q)


	def calc_original_cluster(self):
		self.ASC.set_values('q', self.q)
		self.ASC.set_values('C_num', self.orig_c_num)
		self.ASC.set_values('sigma', self.orig_sigma)
	
		self.ASC.set_values('kernel_type','Gaussian Kernel')
		self.ASC.run()
		self.orig_label = self.ASC.db['allocation']

		#print self.orig_label
		#import pdb; pdb.set_trace()

	def predefine_orig_clustering(self):
		db = self.ASC.db

		self.ASC.set_values('q', self.q)
		self.ASC.set_values('C_num', self.orig_c_num)
		self.ASC.set_values('sigma', self.orig_sigma)
	
		db['Y_matrix'] = Allocation_2_Y(self.orig_label)
		db['prev_clust'] = 1
		db['allocation'] = self.orig_label



	def calc_alt_cluster(self):
		db = self.ASC.db


#		[lambdaV, sigma] = get_lambda_sigma(db)
#		self.median_pair_dist = sigma
#		self.hsic_ratio = lambdaV
#		
#		self.sigma_used = self.sigma_ratio*sigma
#		self.lambda_used = self.lambda_ratio*lambdaV

		self.ASC.set_values('q',self.q)
		self.ASC.set_values('C_num', self.c_num)
#		self.ASC.set_values('sigma',self.sigma_used)
#		self.ASC.set_values('lambda',self.lambda_used)
	
		if self.forced_lambda != None:
			self.ASC.set_values('lambda',self.forced_lambda)
			self.lambda_used = self.forced_lambda
		if self.forced_sigma != None:
			self.ASC.set_values('sigma',self.forced_sigma)
			self.sigma_used = self.forced_sigma
	
		start_time = time.time() 
		self.ASC.run()
		db['run_alternative_time'] = time.time() - start_time
		self.alt_allocation = db['allocation']	
		self.U = db['U_matrix']
		self.Y = db['Y_matrix']
		self.H = db['H_matrix']
		self.W = db['W_matrix']
		self.db = db

	

	def output_alt_info(self):
#
#	
#		db = self.ASC.db
#		print ':::::   ' , db['cf'].calc_cost_function(db['W_matrix'])
#		import pdb; pdb.set_trace()
#
#
		db = self.ASC.db


		against_alternative = np.around(normalized_mutual_info_score(self.alt_label, self.alt_allocation), 3)
		against_truth = np.round(normalized_mutual_info_score(self.alt_allocation, self.orig_label),3)
	
		new_cost = db['cf'].Final_calc_cost(db['W_matrix'])
		CQ = db['cf'].cluster_quality(db)
		Altness = db['cf'].alternative_quality(db)
		[Lagrange_Gradient, new_grad_mag, constraint_mag] = db['cf'].derivative_test(db)


		out_str  = "--- " +			 					str(db['run_alternative_time'] 	)	+	 " seconds ---" +	'\n'
		out_str += '\tq used : ' +			 			str(self.q 						)	+	 '\n'
		out_str += '\torig num clusters :' +				str(self.orig_c_num 			)	+	 '\n'
		out_str += '\talt num clusters :' +				str(self.c_num 					)	+	 '\n'
		out_str += '\tsigma used : '+			 			str(self.sigma_used  			)	+	 '\n'
		#out_str += '\tsigma_ratio : ' +			 		str(self.sigma_ratio 			)	+	 '\n'
		out_str += '\tmedian of pairwise distance : ' + 	str(self.median_pair_dist 		)	+	 '\n'
		out_str += '\tlambda used : '+			 		str(self.lambda_used 			)	+	 '\n'
		#out_str += '\tlambda_ratio : ' +			 		str(self.lambda_ratio 			)	+	 '\n'
		#out_str += '\tHSIC ratio : ' +			 		str(self.hsic_ratio 			)	+	 '\n'
		out_str += '\tAlt vs Alt NMI : ' +			 	str(against_alternative 		)	+	 '\n'
		out_str += '\tAlt vs Orig NMI : ' +			 	str(against_truth 				)	+	 '\n'
		out_str += '\tCost : ' +			 				str(new_cost 					)	+	 '\n'
		out_str += '\tCQ : ' +			 				str(CQ 							)	+	 '\n'
		out_str += '\tAltness : ' +			 			str(Altness 					)	+	 '\n'
		out_str += '\tLagrange Gradient : ' +			str(Lagrange_Gradient 					)	+	 '\n\n'

		out_str += '\tRun Hash : ' 				+ str(db['run_hash'] 	) + '\n'


		#	Cut and Paste
		out_str += '\t:::::::::: Cut and Paste :::::::\n'
		out_str += '\tBest\tMNI\tCQ\tAlt\tCost\tTime\tNMI\tCost\tTime\n'
		out_str += '\t' +  str(np.round(against_alternative , 3)) 
		out_str += '\t' +  str(np.round(against_alternative , 3)) 
		out_str += '\t' +  str(np.round(CQ, 3)  ) 
		out_str += '\t' +  str(np.round(against_truth, 3) ) 
		out_str += '\t' +  str(np.round(new_cost, 3)) 
		out_str += '\t' +  str(np.round(db['run_alternative_time'], 3)) 
		out_str += '\t' + str(np.round(against_alternative,3)) + '\t' + str(np.round(new_cost,3)) + '\t' + str(np.round(db['run_alternative_time'], 3)) + '\n'


		self.write_out(out_str)
		#print self.alt_allocation


		#K = db['cf'].create_Kernel(db['W_matrix'])
		#W = db['W_matrix']
		#U = db['U_matrix']
		#print U
		#H = db['H_matrix']
		#D = db['D_matrix']
		#Y = db['Y_matrix']
		#import pdb; pdb.set_trace()

	
	
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
				

	def maintain_average(self, avg_dict):
		db = self.ASC.db
	
		self.alt_allocation = db['allocation']	
		try:
			alt_alt = np.around(normalized_mutual_info_score(self.alt_label, self.alt_allocation), 3)
			alt_orig = np.round(normalized_mutual_info_score(self.alt_allocation, self.orig_label),3)
		except:
			alt_alt = 0
			alt_orig = 0

		new_cost = db['cf'].Final_calc_cost(db['W_matrix'])
		CQ = db['cf'].cluster_quality(db)
		Altness = db['cf'].alternative_quality(db)

		avg_dict['Alt_V_Alt_NMI'].append(alt_alt)
		avg_dict['Alt_Vs_Orig_NMI'].append(alt_orig)
		avg_dict['Alt'].append(Altness)
		avg_dict['CQ'].append(CQ)
		avg_dict['Cost'].append(new_cost)
		avg_dict['Time'].append(db['run_alternative_time'])



	def save_ISM_init_to_file(self):
		self.set_up_class(False, False )								
		if self.use_predefined_init_clustering:                                    
			self.predefine_orig_clustering()
		else:
			self.calc_original_cluster()                                    
                                                                        
		self.ASC.set_values('save_ISM_init_to_file', True)              
		self.ASC.set_values('W_opt_technique', 'ISM')                   
                                                                        
		self.calc_alt_cluster()                                         
                                                                        
	def plot_result(self):
		pass 

	def perform_default_run(self, debug_info=False):
		self.write_out(':::::::: ISM default RUN  at ' + self.experiment_name + ' ::::::::\n')

		self.set_up_class(debug_info, debug_info)					
		self.ASC.set_values('init_W_from_pickle',False)                 


		if self.use_predefined_init_clustering:                                    
			self.predefine_orig_clustering()
		else:
			self.calc_original_cluster()                                                                              
                                                                        
		self.calc_alt_cluster()                                        
		self.output_alt_info()
		self.save_result_matrices()

	def run_with_W_0(self, technique='SM', pickle_count=0, debug_info=False):
		self.write_out(':::::::: RUN  with W0 at ' + self.experiment_name + ' using ' + technique + ' ::::::::\n')
		self.set_up_class(debug_info, debug_info)
	
		if self.use_predefined_init_clustering:                                    
			self.predefine_orig_clustering()
		else:
			self.calc_original_cluster()

		self.ASC.set_values('W_opt_technique', technique)
		self.ASC.set_values('init_W_from_pickle',True)
		self.ASC.set_values('pickle_count',pickle_count)		

		self.calc_alt_cluster()
		self.output_alt_info()
		self.save_result_matrices()

		#self.plot_result()
		#import pdb; pdb.set_trace()

	def ran_single_from_pickle_init(self, technique, pinit, debug_info=False):
		T = ':::::::::::::   Pickle Initialization of '+  self.experiment_name + ' , ' + technique + ' , '
		T += str(pinit) + ' :::::::::::::\n'
		print T
																					
		avg_dict = {}                                                               
		avg_dict['Alt_V_Alt_NMI'] = []	                                            
		avg_dict['Alt_Vs_Orig_NMI'] = []                                            
		avg_dict['Alt'] = []                                                        
		avg_dict['CQ'] = []														    
		avg_dict['Cost'] = []                                                       
		avg_dict['Time'] = []                                                      
                                                                                   
		self.set_up_class(debug_info, debug_info)								   
		if self.use_predefined_init_clustering:                                    
			self.predefine_orig_clustering()                                       
		else:                                                                      
			self.calc_original_cluster()                                           
		                                                                           
		self.ASC.set_values('W_opt_technique', technique)                          
		self.ASC.set_values('init_W_from_pickle',True)                             
		self.ASC.set_values('pickle_count',pinit)                                  
		self.calc_alt_cluster()                                                    
		self.maintain_average(avg_dict)

		self.output_random_initialize(avg_dict, technique, pinit)

	def random_initializations(self, n_inits, technique):
		print ':::::::::::::   Random Initialization of '+  self.experiment_name + ' , ' + technique + ' :::::::::::::\n'
		#self.write_out(':::::::::::::   Random Initialization of '+  self.experiment_name + ' , ' + technique + ' :::::::::::::\n')

		debug_info = False
		avg_dict = {}
		avg_dict['Alt_V_Alt_NMI'] = []
		avg_dict['Alt_Vs_Orig_NMI'] = []
		avg_dict['Alt'] = []
		avg_dict['CQ'] = []
		avg_dict['Cost'] = []
		avg_dict['Time'] = []

		for pinit in range(n_inits):
			sys.stdout.write("\rCurrently %dth random run." % (pinit))
			sys.stdout.flush()

			self.set_up_class(debug_info, debug_info)

			if self.use_predefined_init_clustering:                                    
				self.predefine_orig_clustering()                                       
			else:                                                                      
				self.calc_original_cluster()
			
			self.ASC.set_values('W_opt_technique', technique)
			self.ASC.set_values('init_W_from_pickle',True)
			self.ASC.set_values('pickle_count',pinit)
			self.calc_alt_cluster()

			self.maintain_average(avg_dict)
			#self.plot_result()

		self.output_random_initialize(avg_dict, technique, 0)

	def write_out(self, out_str):
		fname = 'experiment_results/' + self.experiment_name + '.txt'
		fin = open(fname,'a')
		fin.write(out_str)
		fin.close()

		print out_str

	def save_result_matrices(self):
		db = self.ASC.db
		fname = "saved_matrices/" + db['Experiment_name'] + ".pk"


		if os.path.exists(fname):
			saved_matrices = pickle.load( open( fname, "rb" ) )
		else:
			saved_matrices = {}


		saved_matrices[db['run_hash']] = {}
		saved_matrices[db['run_hash']]['W'] = db['W_matrix']
		saved_matrices[db['run_hash']]['U'] = db['U_matrix']
		saved_matrices[db['run_hash']]['sigma'] = self.sigma_used
		saved_matrices[db['run_hash']]['lambda'] = self.lambda_used
		saved_matrices[db['run_hash']]['c_num'] = self.c_num
		saved_matrices[db['run_hash']]['allocation'] = db['allocation']

		pickle.dump(saved_matrices, open( fname, "wb" ) )


	def output_random_initialize(self, avg_dict, technique, pinit=0):
		Alt_V_Alt_NMI = np.array(avg_dict['Alt_V_Alt_NMI'])
		Alt_V_orig_NMI = np.array(avg_dict['Alt_Vs_Orig_NMI'])
		CQ = np.array(avg_dict['CQ'])
		Alt = np.array(avg_dict['Alt'])
		Cost = np.array(avg_dict['Cost'])
		Time = np.array(avg_dict['Time'])

		#	Display
		out_str = ':::::::::::::   Pickle Initialization of '+  self.experiment_name + ' , ' + technique + ' , '
		out_str += str(pinit) + ' :::::::::::::\n'
		out_str +=  '\tAlt V Alt : ' 				+ 	str(np.round(Alt_V_Alt_NMI.mean() , 2)) 	+ ' ± ' + 	str(np.round(Alt_V_Alt_NMI.std() , 3) )	+ '\n'
		out_str +=  '\tAlt V Orig NMI : ' 	+ 	str(np.round(Alt_V_orig_NMI.mean() , 2)) 	+ ' ± ' + 	str(np.round(Alt_V_orig_NMI.std() , 3) )	+ '\n'
		out_str += '\tAlt : ' 	+ 	str(np.round(Alt.mean() , 2)) 	+ ' ± ' + 	str(np.round(Alt.std(), 3) 	)	+ '\n'
		out_str += '\tCQ : ' 		+ 	str(np.round(CQ.mean()  , 2))	+ ' ± ' +  	str(np.round(CQ.std(), 3) 	)	+ '\n'
		out_str += '\tCost : ' 	+ 	str(np.round(Cost.mean(), 2)) 	+ ' ± ' + 	str(np.round(Cost.std(), 3) )	+ '\n'
		out_str += '\tTime : ' 	+ 	str(np.round(Time.mean(), 2)) 	+ ' ± ' + 	str(np.round(Time.std(), 3) )	+ '\n'
		out_str += '\tq used : ' 						+ str(self.q 				) + '\n'
		out_str += '\torig num clusters :' 			+ str(self.orig_c_num 		) + '\n'
		out_str += '\talt num clusters :' 			+ str(self.c_num 			) + '\n'
		out_str += '\tsigma used : '					+ str(self.sigma_used  		) + '\n'
		out_str += '\tsigma_ratio : ' 				+ str(self.sigma_ratio 		) + '\n'
		out_str += '\tmedian of pairwise distance : ' + str(self.median_pair_dist ) + '\n'
		out_str += '\tlambda used : '					+ str(self.lambda_used 		) + '\n'
		out_str += '\tlambda_ratio : ' 				+ str(self.lambda_ratio 	) + '\n'
		#out_str += '\tHSIC ratio : ' 					+ str(self.hsic_ratio 		) + '\n'


		#	Cut and Paste
		out_str += '\t:::::  Past into data table ::::::\n'
		out_str += '\t' + str(np.round(Alt_V_Alt_NMI.max() , 2))
		out_str += '\t' + str(np.round(Alt_V_Alt_NMI.mean() , 2))			+ '±' + str(np.round(Alt_V_Alt_NMI.std() , 3)) 
		out_str += '\t' + str(np.round(CQ.mean()  , 2))			+ '±' + str(np.round(CQ.std(), 3)  ) 
		out_str += '\t' + str(np.round(Alt_V_orig_NMI.mean() , 2))		+ '±' + str(np.round(Alt_V_orig_NMI.std(), 3) ) 
		out_str += '\t' + str(np.round(Cost.mean(), 2))			+ '±' + str(np.round(Cost.std(), 3)) 
		out_str += '\t' + str(np.round(Time.mean(), 2))			+ '±' + str(np.round(Time.std(), 3)) + '\n'

		self.write_out(out_str)

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
		print 'There are ', len(init_W) , ' elements saved.'

	def find_sigma_lambda(self):
		lowest_cost = float("inf")

		self.sigma_ratio = 1
		self.lambda_ratio = 1
		for self.sigma_ratio in np.arange(0.05,2, 0.05):
			for self.lambda_ratio in np.arange(0.1, 5, 0.2):
				self.set_up_class()
				db = self.ASC.db
				self.ASC.set_values('init_W_from_pickle',False)
				self.calc_original_cluster()
				self.calc_alt_cluster()

				new_cost = db['cf'].calc_cost_function(db['W_matrix'])
				if new_cost < lowest_cost:
					lowest_cost = new_cost
					print '\n', lowest_cost, self.sigma_ratio, self.lambda_ratio
				else:
					sys.stdout.write("\rsigma : %f , lambda : %f, cost : %f. %f" % (self.sigma_ratio, self.lambda_ratio, new_cost, db['sigma']))
					sys.stdout.flush()


	def check_local_minimum(self, W, count, radius):
		db = self.ASC.db
		db['cf'].check_local_minimum(db['W_matrix'], count, radius)

	def run_all_based_on_W0(self):
		self.perform_default_run()
		self.run_with_W_0('SM', 10)
		#self.run_with_W_0('DG', 10)

	def run_all_random(self):
		self.random_initializations(10, 'ISM')
		self.random_initializations(10, 'SM')
		self.random_initializations(10, 'DG')

