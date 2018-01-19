#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust_gpu import *
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

data = genfromtxt('data_sets/facial_85.csv', delimiter=',')		
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
	ASC.set_values('sigma',np.float64(sigma))
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
	print ':::::   ENTERING ALTERNATIVE :::::::::\n\n'
	#rand_lambda = 3*np.random.random()
	rand_lambda = 0.1 # this one works with KDAC
	#rand_lambda = 1

	ASC.set_values('q',4)
	ASC.set_values('C_num',4)
	ASC.set_values('sigma',np.float64(sigma))
	ASC.set_values('lambda', np.float64(rand_lambda))
	#ASC.set_values('kernel_type','Linear Kernel')
	ASC.set_values('kernel_type','Gaussian Kernel')
	start_time = time.time() 
	ASC.run()
	print("--- Time took : %s seconds ---" % (time.time() - start_time))
	alternative = db['allocation']	
	alternative_Y = Allocation_2_Y(alternative)
	pose_Y = Allocation_2_Y(pose_label)


	against_truth = normalized_mutual_info_score(pose_label, alternative)
	against_alternative = normalized_mutual_info_score(label, alternative)

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



	txt = '\tLabel against alternative : ' + str(against_alternative) + '\n'
	txt += '\tLabel against truth : ' + str(against_truth) + '\n'
	txt += '\tAlternative HSIC % diff from random : ' + str(percent_diff) + '\n'
	txt += '\tPose HSIC % diff from random : ' + str(pose_percent_diff) + '\n'
	txt += '\tLambda used : ' + str(rand_lambda) + '\n'
	txt += '\tLowest Cost : ' + str(db['lowest_cost']) + '\n'

	print txt

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




import pdb; pdb.set_trace()


