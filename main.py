#!/usr/bin/env python

#	This is the main file to run various tests
#	To run a particular test, simply uncomment the import
#	while comment all other import tests



import sys
sys.path.append('./tests')
try: import numpy as np
except: 'numpy is missing from the python library, please import it first'
try: import matplotlib
except: 'matplotlib is missing from the python library, please import it first'
try: import sklearn
except: 'sklearn is missing from the python library, please import it first'
try: import scipy
except: 'scipy is missing from the python library, please import it first'

#	numpy settings
np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)



#	Paper publication experiments / uncomment one experiment to run it
import small
#import data_4
#import large_4_gauss
#import moon
#import moon_N
#import flower
#import face
#import webkb
#import Dimond_Gaussians









#	Deprecated tests, should NOT run them
#import test_1_linear_kernel
#import test_2_gaussian_kernel
#import test_3_polynomial_kernel	
#import test_4_small_gaussian
#import test_5_alternative
#import test_6_alternative
#import test_7_polynomial
#import test_7_polynomial
#import test_8				# Simple data
#import gpu_test_8			# GPU Simple data test
#import test_9				# Four Gaussian
#import gpu_test_9			# GPU Four Gaussian
#import test_10				# moon with no noise
#import gpu_test_10			# GPU moon with no noise
#import test_11				# breast cancer
#import test_12				# facial data
#import gpu_test_12			# GPU facial data
#import test_13				# moon with noise
#import gpu_test_13			# moon with noise
#import gene_data_test
#import test_15			# Echer flower image
#import gpu_test_15		# Echer flower image
#import test_16			# Echer mariposa image
#import test_17				# moon 800 
#import test_18				# moon + noise larger values
#import test_19				# Large Gauss different sample numbers
#import test_20				# very very large dataset moon test
#import test_21				# Four Gaussian 3D 1000 samples
