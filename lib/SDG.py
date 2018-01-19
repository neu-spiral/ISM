#!/usr/bin/python

import numpy as np
import pdb
import time
import csv
import sys
import os
import pickle
from create_y_tilde import *
from create_gamma_ij import *
from StringIO import StringIO
from scipy.optimize import minimize
import power_eig
from CPM import *


class SDG:
	def __init__(self, db, iv, jv):
		self.db = db
		self.N = db['data'].shape[0]
		self.d = db['data'].shape[1]	
		self.iv = iv
		self.jv = jv
		self.sigma2 = np.power(db['sigma'],2)
		self.gamma_array = None

		self.costVal_list = []
		self.gradient_list = []
		self.Wchange_list = []

		self.y_tilde = None
		self.debug_mode = True

	def use_coordinate_power_method(self, A, d):
		cpm = CPM(A, Num_of_eigV=d, style='smallest_first', init_with_power_method=True) #largest_first, dominant_first, smallest_first, least_dominant_first
		return [cpm.eigValues, cpm.eigVect]


	def save_initial_W(self, db):
		W_matrix = db['W_matrix']
		fname = "init_files/init_W_" + db['Experiment_name'] + ".pk"
	
		if os.path.exists(fname):
			init_W = pickle.load( open( fname, "rb" ) )
		else:
			init_W = []
	
		init_W.append(W_matrix)
		pickle.dump( init_W, open( fname, "wb" ) )
		print 'There are ' , len(init_W) , ' elements saved.'


	def run_debug_1(self, new_gradient_mag, new_cost, lowest_cost, exit_condition):
		if not self.db['run_debug_1']: return

		if self.debug_mode:
			self.costVal_list.append(new_cost)
			self.gradient_list.append(new_gradient_mag)
			self.Wchange_list.append(exit_condition)
	
			print 'Sum(Aw) : ' , new_gradient_mag, 'New cost :', new_cost, 'lowest Cost :' , lowest_cost, 'Exit cond :' , exit_condition 

	def run_debug_2(self, db, lowest_gradient, lowest_cost):
		if not self.db['run_debug_2']: return

		if self.debug_mode:
			print 'Best : '
			print 'Cost  ' , lowest_cost

			self.db['debug_costVal'].append(self.costVal_list)
			self.db['debug_gradient'].append(self.gradient_list)
			self.db['debug_debug_Wchange'].append(self.Wchange_list)


	def check_positive_hessian(self, w):
		Hessian = np.zeros((self.d, self.d))

		for i in self.iv:
			for j in self.jv:

				gamma_ij = self.create_gamma_ij(self.db, i, j)/self.sigma2
				A_ij = self.create_A_ij_matrix(i,j)
				exponent_term = np.exp(-0.5*w.dot(A_ij).dot(w)/self.sigma2)

				p = A_ij.dot(w)
				Hessian += (gamma_ij/self.sigma2)*exponent_term*(A_ij - (1/self.sigma2)*p.dot(p.T))
	
	
		[eU,eigV,eV] = np.linalg.svd(Hessian)
		
		if np.min(eigV) > 0:
			print 'Positive'
		else:
			print 'Negative'
			pass

	def update_best_W(self, new_cost, new_gradient_mag, W):
		db = self.db

		if db['ISM_always_update']:
			#	Trying always update
			db['lowest_U'] = db['U_matrix']
			db['lowest_cost'] = new_cost
			db['lowest_gradient'] = new_gradient_mag
			db['W_matrix'] = W
		else:
			if(new_cost < db['lowest_cost']):
				db['lowest_U'] = db['U_matrix']
				db['lowest_cost'] = new_cost
				db['lowest_gradient'] = new_gradient_mag
				db['W_matrix'] = W
				#import pdb; pdb.set_trace()




	def run(self):
		db = self.db
		exponent_term = 1
		W = db['W_matrix']

		new_cost = float("inf")
		W_hold = W
		exit_count = db['ISM_exit_count']
		if db['auto_adjust_lambda']:
			exit_count = 30

		for m in range(exit_count):
			[cost, matrix_sum] = db['cf'].calc_cost_function(W, also_calc_Phi=True)

			if db['use_power_method'] and matrix_sum.shape[0] > 500 and m > 0:
				#start_time = time.time() 
				[S2, U2] = self.use_coordinate_power_method(matrix_sum, 1)
				#print 'Power : ' , time.time() - start_time
			else:
				#start_time = time.time() 
				[S2,U2] = np.linalg.eigh(matrix_sum)
				#print 'Eigen : ' , time.time() - start_time


#			start_time = time.time() 
#			[S2, U2] = self.use_coordinate_power_method(matrix_sum, 1)
#			print 'Power : ' , time.time() - start_time
#
#			start_time = time.time() 
#			[S2,U2] = np.linalg.eigh(matrix_sum)
#			print 'Eigen : ' , time.time() - start_time







			
			#import pdb; pdb.set_trace()
			#[S2, U2] = self.use_coordinate_power_method(matrix_sum, 1)
			#[S2,U2] = np.linalg.eigh(matrix_sum)


			eigsValues = S2[0:db['q']]

			new_gradient = matrix_sum.dot(W)	
			Lagrange_gradient = new_gradient - W*eigsValues
			new_gradient_mag = np.linalg.norm(Lagrange_gradient)	




			if False:# switch q in each term, but it doesn't work cus i need to find the differential of W but size is diff
				amax = np.argmax(np.absolute(np.diff(S2))) + 1
				eigsValues = S2[:amax]
				W = U2[:,0:amax]
				print S2
				print 'Amax : ' , amax
			else:
				W = U2[:,0:db['q']]




			if db['save_ISM_init_to_file']:
				db['W_matrix'] = W
				self.save_initial_W(db)
				sys.exit()



			new_cost = db['cf'].calc_cost_function(W)
			if db['auto_adjust_lambda']: db['cf'].balance_lambda(db)	# this will cause the code to run slower, but find out lambda for you

			exit_condition = np.linalg.norm(W - W_hold)/np.linalg.norm(W)
			self.update_best_W(new_cost, new_gradient_mag, W)


			self.run_debug_1(new_gradient_mag, new_cost, db['lowest_cost'], exit_condition)
			if exit_condition < 0.0001: break;
			W_hold = W


		#self.run_debug_2(db, db['lowest_gradient'], Lowest_cost)
		self.run_debug_2(db, db['lowest_gradient'], db['lowest_cost'])
		db['cf'].create_Kernel(db['W_matrix']) # make sure K and D are updated
		return db['W_matrix']


def W_optimize_Gaussian_SDG(db):
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])

	sdg = SDG(db, iv, jv)

	if db['run_W_only_and_Stop']:
		start_time = time.time() 
		sdg.run()
		db['run_W_only_time'] = time.time() - start_time
	else:
		sdg.run()

	

def test_1():		# optimal = 2.4309
	db = {}
	db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1]])
	db['W_matrix'] = np.array([[1,0],[1,1],[0,0]])
	db['sigma'] = 1
	db['lowest_cost'] = float("inf")
	db['lowest_gradient'] = float("inf")

	iv = np.array([0])
	jv = np.array([1,2])
	sdg = SDG(db, iv, jv)
	sdg.gamma_array = np.array([[0,1,2]])
	print sdg.run()
	
	print sdg.calc_cost_function(sdg.W)

	pdb.set_trace()


def test_2():
	q = 4		# the dimension you want to lower it to

	fin = open('data_1.csv','r')
	data = fin.read()
	fin.close()

	db = {}
	db['data'] = np.genfromtxt(StringIO(data), delimiter=",")
	db['N'] = db['data'].shape[0]
	db['d'] = db['data'].shape[1]
	db['q'] = q
		
	db['SGD_size'] = db['N']
	db['sigma'] = 1
	db['lowest_cost'] = float("inf")
	db['lowest_gradient'] = float("inf")

	iv = np.arange(db['N'])
	jv = np.arange(db['N'])
	db['W_matrix'] = np.random.normal(0,10, (db['d'], db['q']) )


	sdg = SDG(db, iv, jv)
	sdg.gamma_array = np.array([[0,1,2,1,1,2], [3,1,3,4,0,2], [1,2,3,8,5,1], [1,2,3,8,5,1], [1,0,0,8,0,0], [1,2,2,1,5,0]])
	#sdg.gamma_array = 4*np.random.rand(6,6)
	sdg.run()
		

	print sdg.W
	print sdg.calc_cost_function(sdg.W)

	import pdb; pdb.set_trace()

#test_2()
