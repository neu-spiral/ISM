
import numpy as np
import sys
from optimize_linear_kernel import *
from optimize_gaussian_kernel import *
from normalize_each_U_row import *
from optimize_polynomial_kernel import *
from K_means import *
from numpy import genfromtxt
from cost_function import *
import time 


class alt_spectral_clust:
	def __init__(self, data_set):
		self.db = {}
		self.db['C_num'] = 2				# Number of clusters

		if type(data_set) == type({}):
			data_set = np.array(data_set)

		data_dimension = data_set.shape
		self.db['N'] = data_dimension[0]
		self.db['d'] = data_dimension[1]
		
		self.db['sigma'] = 1
		self.db['poly_order'] = 2
		self.db['q'] = 1
		self.db['lambda'] = 1
		self.db['auto_adjust_lambda'] = False
		self.db['alpha'] = 0.01
		self.db['SGD_size'] = 10
		self.db['polynomial_constant'] = 1
		self.db['W_opt_technique'] = 'ISM'
		self.db['init_W_from_pickle'] = False
		self.db['pickle_count'] = 0
		self.db['ISM_exit_count'] = 10
		self.db['ISM_always_update'] = True

		self.db['Kernel_matrix'] = np.zeros((self.db['N'],self.db['N']))
		self.db['prev_clust'] = 0
		self.db['Y_matrix'] = np.array([])
		self.db['kernel_type'] = 'Gaussian Kernel'
		#self.db['kernel_type'] = 'Linear Kernel'
		self.db['data_type'] = 'Feature Matrix'

		#outputs from U_optimize
		self.db['D_matrix'] = np.array([])
		self.db['U_matrix'] = np.array([])	
		self.db['W_matrix'] = np.array([])
		self.db['start_time'] = time.time() 

		# output from spectral clustering
		self.db['allocation'] = np.array([])
		self.db['binary_allocation'] = np.array([[0,2,0],[8,2,0]])

		self.db['maximum_W_update_count'] = 200
		self.db['maximum_U_update_count'] = 20
		self.db['data'] = data_set
		self.db['run_hash'] = '0'
		
		self.db['H_matrix'] = np.eye(self.db['N']) - np.ones((self.db['N'],self.db['N']))/self.db['N']
		self.db['cf'] = cost_function(self.db)

		# debug db
		self.db['debug_costVal'] = []
		self.db['debug_gradient'] = []
		self.db['debug_debug_Wchange'] = []
		self.db['save_ISM_init_to_file'] = False

		self.db['run_debug_1'] = True
		self.db['run_debug_2'] = True


		self.db['run_W_only_and_Stop'] = False
		self.db['run_W_only_time'] = 0
		self.db['use_power_method'] = False


#	def initialize_U(self, db):
#		properly_initialize_U(db)

	def remove_previous_Y_columns(self):
		db = self.db
		Y = db['Y_matrix']
		k = db['C_num']
		db['Y_matrix'] = Y[:,0:Y.shape[1] - k]


	def set_values(self, key, val):
		self.db[key] = val
	
	def center_data(self, data_set):
		return np.dot(self.db['H_matrix'], data_set)

	def run(self):
		db = self.db
		N = self.db['N']

		if db['data_type'] == 'Feature Matrix': 
			self.db['data'] = self.center_data(self.db['data'])

		if self.db['kernel_type'] == 'Linear Kernel':
			optimize_linear_kernel(self.db)
		elif self.db['kernel_type'] == 'Gaussian Kernel':
			optimize_gaussian_kernel(self.db)
		elif self.db['kernel_type'] == 'Polynomial Kernel':
			optimize_polynomial_kernel(self.db)
		else :
			raise ValueError('Error : unknown kernel was used.')
	
		normalize_each_U_row( self.db )
		K_means(self.db)	
		self.db['prev_clust'] += 1

