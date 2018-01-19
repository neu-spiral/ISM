#!/usr/bin/python

from numpy import *

class orthogonal_optimization:
	def __init__(self, cost_function, gradient_function):
		self.cost_function = cost_function
		self.gradient_function = gradient_function
		self.x_opt = None
		self.cost_opt = None
		self.db = None

	def calc_A(self, x):
		G = self.gradient_function(x)
		A = G.dot(x.T) - x.dot(G.T)
		return A


	def run(self, x_init, max_rep=200):
		d = x_init.shape[0]
		self.x_opt = x_init
		I = eye(d)
		converged = False
		x_change = linalg.norm(x_init)
		m = 0

		while( (converged == False) and (m < max_rep)):
			alpha = 2
			cost_1 = self.cost_function(self.x_opt)
			A = self.calc_A(self.x_opt)

			while(alpha > 0.000000001):
				next_x = linalg.inv(I + alpha*A).dot(I - alpha*A).dot(self.x_opt)
				cost_2 = self.cost_function(next_x)
	
				if self.db != None:
					Alt = self.db['cf'].alternative_quality(self.db)
					if self.db['run_debug_1']: print alpha, cost_1, cost_2, 'Alt : ' , Alt

				if((cost_2 < cost_1) or (abs(cost_1 - cost_2)/abs(cost_1) < 0.0000001)):
					x_change = linalg.norm(next_x - self.x_opt)
					[self.x_opt,R] = linalg.qr(next_x)		# QR ensures orthogonality
					self.cost_opt = cost_2
					break
				else:
					alpha = alpha*0.2

			m += 1

			if(x_change < 0.00001*linalg.norm(self.x_opt)): converged = True

		return self.x_opt	
