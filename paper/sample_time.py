#!/usr/bin/env python

import numpy as np
import pickle
import matplotlib.pyplot as plt


def dimension_plot():
	time_info = pickle.load(open('Size_Dim_V_time.pk','r'))


	X1 = time_info['Dim_V_time_True']['log_num_samples']
	Y1 = time_info['Dim_V_time_True']['log_time']
	X1 = X1[5:]
	Y1 = Y1[5:]

	X2 = time_info['Dim_V_time_False']['log_num_samples']
	Y2 = time_info['Dim_V_time_False']['log_time']
	X2 = X2[5:]
	Y2 = Y2[5:]


	coef = np.polyfit(X1, Y1, 1)
	p = np.poly1d(coef)
	S1 = coef[0]
	Y1_B = p(X1)

	coef = np.polyfit(X2, Y2, 1)
	S2 = coef[0]
	p = np.poly1d(coef)
	Y2_B = p(X2)

	ax = plt.gca()
	
	labels = [0]
	for m in range(len(X1)):
		v = '$2^{' + str(X1[m]) + '}$'
		print v
		labels.append(v)

	ax.set_xticklabels(labels)

	labels = [0]
	for m in range(4,14):
		#vv = m*2 - 2
		v = '$2^{' + str(m) + '}$'
		print v
		labels.append(v)
	ax.set_yticklabels(labels)


	plt.subplot(111)
	plt.plot(X1, Y1, 'go')
	plt.plot(X2, Y2, 'bo')

	line_up, = plt.plot(X1, Y1_B, 'g', label='Power Method, slope : ' + str(np.round(S1,2)) )
	line_down, = plt.plot(X2, Y2_B, 'b--', label='Eigen Decomp, slope : ' + str(np.round(S2,2)))

	plt.legend(handles=[line_up, line_down])
	plt.xlabel('Number of dimensions')
	plt.ylabel('Run Time(s)')
	plt.title('Num of Dimensions Vs Time in log2/log2 scale')
	plt.show()









#	#N = [64, 128, 256]
#	N = [256]
#	Power = [True, False]
#
#	time_info['Dim_V_time_False']
#
#	import pdb; pdb.set_trace()	
#
#	plt.figure(1)
#	label_list = []
#	plt.subplot(111)
#	for n in N:
#		for p in Power:
#			keyV = 'Dim_V_time_N' + str(n) + '_' + str(p)
#			X = time_info[keyV]['log_num_samples']
#			Y = time_info[keyV]['log_time']
#			
#			coef = np.polyfit(X, Y, 1)
#			p = np.poly1d(coef)
#			Yb = p(X)
#	
#			pltLable, = plt.plot(X, Y, 'o', label=keyV)
#			pltLable, = plt.plot(X, Yb, label=keyV)
#			label_list.append(pltLable)
#	
#	
#	#plt.xlabel('Feature 1')
#	#plt.ylabel('Feature 2')
#	#plt.title('original plot')
#	
#	plt.legend(handles=label_list)
#	plt.show()
#	
#	
#	#print time_info.keys()
#	import pdb; pdb.set_trace()

def Sample_plot():
	time_info = pickle.load(open('Size_Dim_V_time.pk','r'))
	#D = [64, 128, 256]
	D = [256]
	Power = [False]
	
	plt.figure(1)
	label_list = []
	plt.subplot(111)
	for d in D:
		for p in Power:
			keyV = 'Dim_V_time_N' + str(d) + '_' + str(p)

			X = time_info[keyV]['log_num_samples']
			Y = time_info[keyV]['log_time']
			#Y = time_info[keyV]['time_in_s']
			
			coef = np.polyfit(X, Y, 1)
			p = np.poly1d(coef)
			Yb = p(X)
	
			lb = 'Dimension : ' + str(d) + ' , slope : ' + str(np.round(coef[0],2))
			

			pltLable, = plt.plot(X, Y, 'o')
			pltLable, = plt.plot(X, Yb, label=lb)
			label_list.append(pltLable)
	

	ax = plt.gca()
	labels = [0]
	for m in range(len(X)):
		v = '$2^{' + str(X[m]) + '}$'
		labels.append(v)
	ax.set_xticklabels(labels)

	#Y_Tic = ax.get_yticklabels()
	#for p in Y_Tic: print p

	labels = [0]
	for m in range(0,8):
		vv = m*2 - 2
		v = '$2^{' + str(vv) + '}$'
		print v
		labels.append(v)
	ax.set_yticklabels(labels)


	#ax.set_xscale("log", nonposx='clip')
	#ax.set_yscale("log", basey=2)



	plt.xlabel('Number of samples')
	plt.ylabel('Run Time(s)')
	plt.title('Num of Samples Vs Time in log2/log2 scale')	
	
	plt.legend(handles=label_list)
	plt.show()
	
	
	#print time_info.keys()
	import pdb; pdb.set_trace()

#Sample_plot()
dimension_plot()
