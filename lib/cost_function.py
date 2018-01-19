#!/usr/bin/python

import numpy as np
import pdb
import csv
import sys
import sklearn.metrics
from create_y_tilde import *
from create_gamma_ij import *
import matplotlib.pyplot as plt
from U_optimize import *
from StringIO import StringIO
from scipy.optimize import minimize
from sklearn.kernel_approximation import RBFSampler
import time 

#import autograd.numpy as np
#from autograd import grad

## Define a function Tr(WTA W), we know that gradient = (A+AT)W
#def cost_foo(W, db): 
#	K = db['Kernel_matrix']
#	U = db['U_matrix']
#	H = db['H_matrix']
#	D = db['D_matrix']
#	l = db['lambda']
#	Y = db['Y_matrix']
#
#	s1 = np.dot(np.dot(D,H),U)
#	HSIC_1 = np.dot(s1, np.transpose(s1))*K
#
#	s2 = np.dot(np.dot(D,H),Y)
#	HSIC_2 = l*np.dot(s2, np.transpose(s2))*K
#
#	return np.sum(HSIC_2 - HSIC_1)
#
#grad_foo = grad(cost_foo)       # Obtain its gradient function

class cost_function:
	def __init__(self, db):
		self.db = db
		self.N = db['data'].shape[0]
		self.d = db['data'].shape[1]

		self.iv = np.array(range(self.N))
		self.jv = self.iv


		self.y_tilde = None
		self.W = None
		self.gamma = np.zeros((self.N, self.N))
		self.exp = np.zeros((self.N, self.N))
		self.gamma_exp = np.empty((self.N, self.N))
		self.A_memory_feasible = True

		self.psi = None			# This is the middle term that doesn't change unless U or Y update
		self.Q = None			# This is the tensor term of ( X tensor 1 ) - (1 tensor X)

		#try:
		#	self.A = np.empty((self.N,self.N,self.d, self.d))
		#	self.Aw = np.empty((self.N,self.N,self.d, self.q))
		#	self.create_A()
		#except:
		#	self.A_memory_feasible = False
		#	raise
	
		self.calc_Q()

	def check_local_minimum(self, W, counter=200, radius=1):
		db = self.db
		lowest_cost = self.Final_calc_cost(W)
		print 'Original Lowest cost :' , lowest_cost, ' Derivative : ' , self.derivative_test(db, W)

		for mo in range(counter):
			sys.stdout.write("\rAt %f" % (mo))
			sys.stdout.flush()

			#W_temp = W + radius*np.random.randn(db['d'], db['q']) 			# randomize initialization
			W_temp = radius*np.random.randn(db['d'], db['q']) 			# randomize initialization
			[Q,R] = np.linalg.qr(W_temp)
			new_cost = self.Final_calc_cost(Q)
			if new_cost < lowest_cost:
				lowest_cost = new_cost
				print 'Lower Cost : ' , new_cost , ' Derivative : ' , self.derivative_test(db, Q)


	def calc_Q(self):
		try:
			X = self.db['data']
			one_vector = np.ones((self.N,1))
			self.Q = np.kron(X,one_vector) - np.kron(one_vector, X)
		except:
			self.A_memory_feasible = False
	
	def init_ISM_matrices(self, W):
		self.create_Kernel(W)
		U_optimize(self.db)

	def calc_psi(self, Y_columns=None): # psi = H(UU'-l YY')H
		db = self.db

		if(Y_columns != None): 
			Y = db['Y_matrix'][:,0:Y_columns]
		else:
			if self.db['prev_clust'] == 0: 
				Y = db['Y_matrix']
			else:
				#num_of_columns = self.db['prev_clust']*db['q']
				num_of_columns = db['q']
				Y = db['Y_matrix'][:,0:num_of_columns]


		U = db['U_matrix']
		H = db['H_matrix']

		self.psi = H.dot(U.dot(U.T) - db['lambda']*Y.dot(Y.T)).dot(H)
		return self.psi

	def create_D_matrix(self, kernel):
		d_matrix = np.diag(1/np.sqrt(np.sum(kernel,axis=1))) # 1/sqrt(D)
		return d_matrix


	def create_Kernel(self, W):
		db = self.db
		sigma = db['sigma']
		
		X = db['data'].dot(W)
		gamma_V = 1.0/(2*np.power(sigma,2))

		if self.N < 30000:	# if rbf kernel too large use RFF instead
			db['Kernel_matrix'] = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma_V)
		else:
			rbf_feature = RBFSampler(gamma=gamma_V, random_state=1, n_components=2000)
			Z = rbf_feature.fit_transform(X)
			db['Kernel_matrix'] = Z.dot(Z.T)

		db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)

		return db['Kernel_matrix']


	def test_2nd_order(self, db, W=None):	# Test the derivative at a point
		if type(W) == type(None): W = db['W_matrix']
		H = db['H_matrix']
		U = db['U_matrix']
		Y = db['Y_matrix']

		sigma2 = np.power(db['sigma'],2)
		Ku = U.dot(U.T)
		Ky = Y.dot(Y.T)
		suggest_lambda = np.trace(Ky.T.dot(Ku))/np.trace(Ky.T.dot(Ky))

		K = self.create_Kernel(W)
		D = np.diag(db['D_matrix'])
		DD = np.outer(D,D)
		const_matrix = DD*self.psi*K/sigma2

		const_matrix_2 = DD*np.absolute(self.psi)*K
		RHS = np.sum(const_matrix_2)/sigma2
		#RHS = np.sum(const_matrix)/sigma2

		A = np.zeros((self.d, self.d))
		#RHS = 0
		for i in self.iv:
			for j in self.jv:
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
				A_ij = np.dot(x_dif.T, x_dif)
				A += const_matrix[i,j]*A_ij
				#RHS += const_matrix_2[i,j]*np.trace(A_ij.dot(A_ij))

				#AA = A_ij*A_ij
				#print AA
				#import pdb; pdb.set_trace()

		[S2,U2] = np.linalg.eigh(A)

		LHS = (S2[db['q']] - S2[db['q']-1])*sigma2
		print '::::::::::: 2nd order study :::::'
		print S2	
		print 'Differential : ', (S2[db['q']] - S2[db['q']-1])
		print 'Suggest Lambda: ', suggest_lambda
		print LHS , ' > ' , RHS

		
		plt.figure(1)	
		plt.subplot(111)
		plt.plot(S2, 'bo')
		plt.xticks(np.arange(0, len(S2)+1, 1.0))

		plt.title(db['Experiment_name'])
		plt.show()
		
		return 


	def derivative_test(self, db, W=None):	# Test the derivative at a point
		if type(W) == type(None): W = db['W_matrix']

		K = self.create_Kernel(W)
		H = db['H_matrix']
		U = db['U_matrix']

		K = self.create_Kernel(W)
		D = np.diag(db['D_matrix'])
		DD = np.outer(D,D)
		const_matrix = DD*self.psi*K/np.power(db['sigma'],2)
	

		A = np.zeros((self.d, self.d))
		for i in self.iv:
			for j in self.jv:
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
				A_ij = np.dot(x_dif.T, x_dif)
				A += const_matrix[i,j]*A_ij

		if True:# Use eig
			[S2,U2] = np.linalg.eigh(A)
			eigsValues = S2[0:db['q']]
			#W = U2[:,0:db['q']]
		else:
			# Use svd
			[U,S,V] = np.linalg.svd(A)
			reverse_S = S[::-1]
			eigsValues = reverse_S[0:db['q']]
			#W = np.fliplr(U)[:,0:db['q']]

		new_gradient = A.dot(W)

		constraint_mag = np.linalg.norm(W*eigsValues)
		new_grad_mag = np.linalg.norm(new_gradient)

		Lagrange_gradient = new_gradient - W*eigsValues
		lagrange_gradient_mag = np.linalg.norm(Lagrange_gradient)
		return [lagrange_gradient_mag, new_grad_mag, constraint_mag]

	def balance_lambda(self, db):
		cq = self.cluster_quality(db)
		aq = self.alternative_quality(db)
		db['lambda'] = np.abs(cq/aq)
		print 'Lambda : ' , db['lambda']

	def cluster_quality(self, db):
		W = db['W_matrix']
		K = self.create_Kernel(W)
		D = db['D_matrix']
		H = db['H_matrix']
		U = db['U_matrix']

		return np.trace( H.dot(D).dot(K).dot(D).dot(H).dot(U).dot(U.T) ) 

	def alternative_quality(self, db):
		W = db['W_matrix']
		K = self.create_Kernel(W)
		D = db['D_matrix']
		H = db['H_matrix']
		Y = db['Y_matrix']

		if db['prev_clust'] == 0: Y = db['Y_matrix']
		else:
			#num_of_columns = db['prev_clust']*db['q']
			num_of_columns = db['q']
			Y = db['Y_matrix'][:,0:num_of_columns]


		return np.trace( H.dot(D).dot(K).dot(D).dot(H).dot(Y).dot(Y.T) ) 

	def calc_gradient_function(self, W):
		self.psi = None
		[cost, Phi] = self.calc_cost_function(W, also_calc_Phi=True)
		gradient = Phi.dot(W)
		return gradient

	def Final_calc_cost(self, W): 
		self.psi = None
		return self.calc_cost_function(W, Y_columns= self.db['q'])

	def calc_cost_function(self, W, also_calc_Phi=False, update_D_matrix=False, Y_columns=None): #Phi = the matrix we perform SVD on
		db = self.db
		if type(self.psi) == type(None): self.calc_psi(Y_columns)
		self.sigma2 = np.power(db['sigma'],2)
	
		#start_time = time.time() 
		K = self.create_Kernel(W)
		D = np.diag(db['D_matrix'])
		DD = np.outer(D,D)

		const_matrix = DD*self.psi*K
		cost = -np.sum(const_matrix)


		if not also_calc_Phi: return cost
		if self.A_memory_feasible:
			O = np.reshape(const_matrix, (1,const_matrix.size))
			Phi = ((self.Q.T*O).dot(self.Q))/self.sigma2
			return [cost, Phi]
		else:
			print '\n\nYou still need to write the part where memory is not feasible.\n\n'
			#	You will have to multiply each A and add them up
			raise



def test_1():		# optimal = 2.4309
	np.set_printoptions(precision=4)
	np.set_printoptions(threshold=np.nan)
	np.set_printoptions(linewidth=300)


	db = {}
	db['data'] = np.array([[5,5],[4,4],[-5,5],[-4,4],[5,-4],[4,-3],[-5,-4],[-4,-3]])
	db['W_matrix'] = np.array([[1],[1]])
	db['sigma'] = 2
	db['lambda'] = 1
	db['N'] = db['data'].shape[0]
	N = float(db['N'])
	db['C_num'] = 2

	db['lowest_cost'] = float("inf")
	db['lowest_gradient'] = float("inf")
	db['Kernel_matrix'] = sklearn.metrics.pairwise.rbf_kernel(db['data'], gamma=(0.5/np.power(db['sigma'],2)))
	db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)
	L = db['D_matrix'].dot(db['Kernel_matrix']).dot(db['D_matrix'])
	[U,S,V] = np.linalg.svd(L)
	db['U_matrix'] = U[:,:db['C_num']]

	db['Y_matrix'] = np.array([[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1]])
	db['H_matrix'] = np.eye(db['N']) - (1.0/N)*np.ones((db['N'], db['N']))

	c_f = cost_function(db)
	print c_f.calc_cost_function(db['W_matrix'])


#test_1()
