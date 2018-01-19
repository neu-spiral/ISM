#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
from numpy import genfromtxt
import numpy.matlib
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import sklearn
import time 
from cost_function import *
import matplotlib.pyplot as plt
from Y_2_allocation import *


print 'Test 14 Web KB data \n\n'

university_labels  = genfromtxt('data_sets/webkb_processed/webkbRaw_label_univ.csv', delimiter=',')
topic_labels  = genfromtxt('data_sets/webkb_processed/webkbRaw_label_topic.csv', delimiter=',')
data = genfromtxt('data_sets/webkb_processed/min_words.csv', delimiter=',')

d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
sigma_value = np.median(d_matrix)

ASC = alt_spectral_clust(data)
db = ASC.db

if False:
	ASC.set_values('q',4)
	ASC.set_values('C_num',4)
	ASC.set_values('sigma',sigma_value)
	ASC.set_values('kernel_type','Gaussian Kernel')
	start_time = time.time() 
	ASC.run()
	a = db['allocation']
	print "University NMI : " , normalized_mutual_info_score(a,university_labels)
	print "topic NMI : " , normalized_mutual_info_score(a, topic_labels)
	print("Time for spectral Clustering : %s seconds ---" % (time.time() - start_time))
else:
	print ':::::   USE PRE-DEFINED CLUSTERING :::::::::\n\n'
	ASC.set_values('kernel_type','Gaussian Kernel')

	db['Y_matrix'] = Allocation_2_Y(university_labels)
	db['prev_clust'] = 1
	db['allocation'] = university_labels


if True:	# run alternative clustering
	#rand_lambda = 3*np.random.random()
	rand_lambda = 0.057 # this one works with FKDAC
	#rand_lambda = 0.25
	#rand_lambda = 1  # this work works with KDAC

	ASC.set_values('q',4)
	ASC.set_values('C_num',4)
	ASC.set_values('sigma',sigma_value)
	ASC.set_values('lambda',rand_lambda)
	ASC.set_values('maximum_W_update_count',30)			# Sets the counter to loop only 3 times
	ASC.set_values('maximum_U_update_count',20)			# Sets the counter to loop only 3 times
	ASC.set_values('Experiment_name','WebKB')
	ASC.set_values('ISM_always_update', True)

	db['W_opt_technique'] = 'ISM'  # DG, SM, or ISM
	#db['init_W_from_pickle'] = True
	#db['pickle_count'] = 0
	ASC.set_values('ISM_exit_count',10)

	#ASC.set_values('kernel_type','Linear Kernel')
	ASC.set_values('kernel_type','Gaussian Kernel')
	start_time = time.time() 

	ASC.run()
	db['run_alternative_time'] = time.time() - start_time
	print("--- Time took : %s seconds ---" % db['run_alternative_time'])
	alternative = db['allocation']	
	against_alternative = normalized_mutual_info_score(university_labels, alternative)
	against_truth = normalized_mutual_info_score(topic_labels, alternative)
	final_cost = db['cf'].calc_cost_function(db['W_matrix'])
	CQ = db['cf'].cluster_quality(db)

	print 'NMI against original :' , against_alternative
	print 'NMI True :' , against_truth
	print 'Cost : ' , final_cost
	print 'CQ : ' , CQ
	print 'Alt : ' , (final_cost - CQ)
	print 'sigma : ' , sigma_value
	


if False:		# test if the converged point is a saddle point
	W = db['W_matrix']
	lowest_W = W.copy()	

	for m in range(1000):
		W_temp = 0.01*np.random.randn(db['d'], db['q']) + W
		[Q,R] = np.linalg.qr(W_temp)

		new_cost = db['cf'].calc_cost_function(Q)
		if new_cost < final_cost:
			print 'lower : ' , new_cost
			final_cost = new_cost
			lowest_W = Q
		else:
			print new_cost, 


if False:		#	Output the result to a text file
	cf = db['cf']
	outLine = str(against_truth) + '\t' + str(against_alternative) + '\t' 
	outLine += str(np.round(cf.cluster_quality(db), 4)) + '\t' + str(np.round(cf.calc_cost_function(db['W_matrix']),3))
	outLine += '\t' + str(np.round(db['run_alternative_time'],3)) + '\n'

	fin = open('webKB_result_DG.txt','a')
	fin.write(outLine)
	fin.close()






#	alternative_Y = Allocation_2_Y(alternative)
#	topic_Y = Allocation_2_Y(topic_labels)
#
#
#	against_truth = normalized_mutual_info_score(topic_labels, alternative)
#	against_alternative = normalized_mutual_info_score(university_labels, alternative)
#
#	alternative_HSIC = HSIC_rbf(data, alternative_Y, sigma_value)
#	pose_HSIC = HSIC_rbf(data, topic_Y, sigma_value)
#
#
#
#	random_HSIC = np.array([])
#	for m in range(10):
#		random_allocation = np.floor(alternative_Y.shape[1]*np.random.random(data.shape[0]))
#		random_Y = Allocation_2_Y(random_allocation)
#		H = HSIC_rbf(data, random_Y, sigma_value)
#		random_HSIC = np.hstack((random_HSIC,H))
#
#	mean_RHSIC = np.mean(random_HSIC)
#	percent_diff = (alternative_HSIC - mean_RHSIC)/alternative_HSIC
#	pose_percent_diff = (pose_HSIC - mean_RHSIC)/pose_HSIC
#
#
#	txt = '\tLabel against alternative : ' + str(against_alternative) + '\n'
#	txt += '\tLabel against truth : ' + str(against_truth) + '\n'
#	txt += '\tAlternative HSIC % diff from random : ' + str(percent_diff) + '\n'
#	txt += '\tPose HSIC % diff from random : ' + str(pose_percent_diff) + '\n'
#	txt += '\tLambda used : ' + str(rand_lambda) + '\n'
#	txt += '\tLowest Cost : ' + str(db['lowest_cost']) + '\n'
#
#	print txt
#
#	#append_txt('./output.txt', txt)
#
#	#print "Alternative aginst original: " , normalized_mutual_info_score(pose_label, alternative)
#	#print "Alternative aginst truth : " , normalized_mutual_info_score(label, alternative)
	

import pdb; pdb.set_trace()

try:
	if True:	# plot the W convergence results
		X = db['data']
		plt.figure(2)
		
		plt.suptitle('WebKB.csv',fontsize=24)
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
except:
	import pdb; pdb.set_trace()

import pdb; pdb.set_trace()
