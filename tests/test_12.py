#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
import numpy.matlib
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import sklearn
import time 
from cost_function import *
import matplotlib.pyplot as plt
from Y_2_allocation import *
from HSIC import *
from file_writing import *



#data = genfromtxt('data_sets/facial_960.csv', delimiter=',')		
#data = genfromtxt('data_sets/facial.csv', delimiter=',')		
#sunglass_label = genfromtxt('data_sets/facial_sunglasses_labels.csv', delimiter=',')		
#pose_label = genfromtxt('data_sets/facial_pose_labels.csv', delimiter=',')		
#original_Y = genfromtxt('data_sets/facial_original_Y.csv', delimiter=',')		
#data = genfromtxt('data_sets/facial_98.csv', delimiter=',')		
#data = genfromtxt('data_sets/facial_95.csv', delimiter=',')		

data = genfromtxt('data_sets/facial_85.csv', delimiter=',')		# 27 samples
label = genfromtxt('data_sets/facial_true_labels_624x960.csv', delimiter=',')		
pose_label = genfromtxt('data_sets/facial_pose_labels_624x960.csv', delimiter=',')		
original_Y = Allocation_2_Y(label)
name_file = open('data_sets/facial_names_624x960.csv', 'r')



#	Use the median of the pairwise distance as sigma
d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
sigma = np.median(d_matrix)
#sigma = 18   # HSIC(true pose,data) = 78.8 


names = np.array(name_file.readlines())
name_file.close()

ASC = alt_spectral_clust(data)
db = ASC.db

if False:	# run original spectral clustering
	ASC.set_values('q',30)
	ASC.set_values('C_num',20)
	ASC.set_values('sigma',sigma)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.run()
	original = db['allocation']
	print "Original Clustering Vs Truth NMI: " , normalized_mutual_info_score(label, original)

else: 		# run preset original clustering
	print ':::::   USE PRE-DEFINED CLUSTERING :::::::::\n\n'
	ASC.set_values('kernel_type','Gaussian Kernel')

	db['Y_matrix'] = original_Y
	db['prev_clust'] = 1
	db['allocation'] = Y_2_allocation(original_Y)
	#a = db['allocation']
	#print 'Predefined allocation :' , a , '\n'




if True:	# run alternative clustering
	#rand_lambda = 3*np.random.random()
	rand_lambda = 0.09 # this one works with FKDAC
	#rand_lambda = 1  # this work works with KDAC

	ASC.set_values('q',4)
	ASC.set_values('C_num',4)
	ASC.set_values('sigma',sigma)
	ASC.set_values('lambda',rand_lambda)
	#ASC.set_values('kernel_type','Linear Kernel')
	ASC.set_values('kernel_type','Gaussian Kernel')
	db['W_opt_technique'] = 'ISM'  # DG, SM, or ISM
	db['Experiment_name'] = 'face'
	ASC.set_values('ISM_exit_count',2)
	ASC.set_values('init_W_from_pickle',False)
	ASC.set_values('pickle_count',0)

	start_time = time.time() 
	ASC.run()

	time_it_took = time.time() - start_time
	print("--- Time took : %s seconds ---" % time_it_took)
	alternative = db['allocation']	
	alternative_Y = Allocation_2_Y(alternative)
	pose_Y = Allocation_2_Y(pose_label)


	against_truth = np.round(normalized_mutual_info_score(pose_label, alternative),3)
	against_alternative = np.round(normalized_mutual_info_score(label, alternative),3)

	alternative_HSIC = HSIC_rbf(data, alternative_Y, sigma)
	pose_HSIC = HSIC_rbf(data, pose_Y, sigma)



	random_HSIC = np.array([])
	for m in range(10):
		random_allocation = np.floor(alternative_Y.shape[1]*np.random.random(data.shape[0]))
		random_Y = Allocation_2_Y(random_allocation)
		H = HSIC_rbf(data, random_Y, sigma)
		random_HSIC = np.hstack((random_HSIC,H))

	mean_RHSIC = np.mean(random_HSIC)
	percent_diff = (alternative_HSIC - mean_RHSIC)/alternative_HSIC
	pose_percent_diff = (pose_HSIC - mean_RHSIC)/pose_HSIC




	cf = db['cf']


	txt = '\tLabel against alternative : ' + str(against_alternative) + '\n'
	txt += '\tLabel against truth : ' + str(against_truth) + '\n'
	txt += '\tAlternative HSIC % diff from random : ' + str(percent_diff) + '\n'
	txt += '\tPose HSIC % diff from random : ' + str(pose_percent_diff) + '\n'
	txt += '\tLambda used : ' + str(rand_lambda) + '\n'
	txt += '\tLowest Cost : ' + str(db['lowest_cost']) + '\n'
	print txt


	outLine = str(against_truth) + '\t' + str(against_alternative) + '\t' 
	outLine += str(np.round(cf.cluster_quality(db), 4)) + '\t' + str(np.round(db['lowest_cost'],3))
	outLine += '\t' + str(time_it_took) + '\n'

	fin = open('face_result.txt','a')
	fin.write(outLine)
	fin.close()

	#append_txt('./output.txt', txt)

	#print "Alternative aginst original: " , normalized_mutual_info_score(pose_label, alternative)
	#print "Alternative aginst truth : " , normalized_mutual_info_score(label, alternative)
	



##print "NMI Against Sunglasses label : " , normalized_mutual_info_score(sunglass_label,alternative)
#
#print names[alternative == 1].shape
#print names[alternative == 2].shape
#print names[alternative == 3].shape
#print names[alternative == 4].shape
#
##for m in names[original== 1]: print m, 
#for m in names[alternative == 1]: print m, 
##for m in names[alternative == 2]: print m, 



if False:	# save or load db to and from a pickle file
	plot_info = {}
	plot_info['debug_costVal'] = db['debug_costVal']
	plot_info['debug_gradient'] = db['debug_gradient']
	plot_info['debug_debug_Wchange'] = db['debug_debug_Wchange']
	pickle.dump( plot_info, open( "tmp_db.pk", "wb" ) )


	#plot_info= pickle.load( open( "tmp_db.pk", "rb" ) )
	#db['debug_costVal'] = plot_info['debug_costVal']
	#db['debug_gradient'] = plot_info['debug_gradient']
	#db['debug_debug_Wchange'] = plot_info['debug_debug_Wchange']



if False:	# plot the W convergence results
	X = db['data']
	plt.figure(2)
	
	plt.suptitle('facial.csv',fontsize=24)
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


import pdb; pdb.set_trace()


