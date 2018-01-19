#!/usr/bin/env python
# -*- coding: UTF-8 -*-

bold = False
#A = '1	1	2	0	-1.2	0.02	1	-1.2	0.02	1	0.04	1'
#A = '1	0.93±0.196	0.99±0.0	0.03±0.104	-0.99±0.0	1.51±0.102	0	-0.991	1.466'


#B =  'ISM	1	1	1.9	0	100.267	0.166	1	100.267	0.166\n'
#B += 'SM	1	0.3±0.458	1.9±0.029	0.6±0.49	227.15±87.974	37.3±47.368	1	100.267	0.32\n'
#B += 'DG	0.1	0.1±0.32	0.61±0.17	0.9±0.32	101.5±30.8	1688±551	1	100.432	1614.817'

#B = 'ISM	1	1	1.9	0	100.267	0.166	1	100.267	0.166\n'
#B += 'SM	1	0.3±0.458	1.9±0.029	0.6±0.49	227.15±87.974	37.3±47.368	1	100.267	0.32\n'
#B += 'DG	0.1	0.1±0.32	0.61±0.17	0.9±0.32	101.5±30.8	1688±551	1	100.432	1614.817\n'

#Moon
#B = 'ISM	1	1	2	0	149.383	0.575	1	149.383	0.575\n'
#B += 'SM	0.73	0.27±0.335	2.0±0.0	0.49±0.488	276.77±18.122	258.47±373.734	1	149.383	2.127\n'
#B += 'DG	0.89	0.09±0.267	1.27±0.553	0.8±0.398	161.65±101.831	3212.2±1368	1	149.697	631.245'

#Moon_N
#B = 'ISM	1	1	2	0	15.351	0.3	1	15.3	0.3\n'
#B += 'SM	1	0.22±0.394	2.0±0.0	0.72±0.427	17.17±1.252	102.12±48.536	1	15.3	0.31\n'
#B += 'DG	0	0.0±0.0	1.63±0.037	1.0±0.0	16.2±0.061	3588.28±304.3	1	15.432	5685.51\n'

#Flower 
#B = 'ISM	0	0	0.413	0	19.514	0.061	0	19.514	0.061\n'
#B += 'SM	0	0	0.3±0.179	0.01±0.012	0.72±0.483	0.81±0.596	0	19.514	0.472\n'
#B += 'DG	0	0	0.35±0.019	0.2±0.385	20.63±1.361	37.03±5.432	0	19.608	58.37\n'

<<<<<<< HEAD
##Face
=======
###Face
>>>>>>> 7fa38d169e52d53e5031bc3ebad54413b4e4c583
#B = 'ISM	0.57	0.57	0.61	0.004	59.7	1.5	0.57	59.7	1.523\n'
#B += 'SM	0.57	0.57±0.002	0.61±0.0	0.0±0.0	59.7±0.131	109±35.187	0.56	59.7	47.55\n'
#B += 'DG	0.458±0.045	0.458±0.045	0.565±0.018	0.024±0.018	54.28±2.97	166070±4342	0.564	59.8	102468'

##Webkb
#B = 'ISM	0.37	0.37	0.286	0	-0.273	231	0.37	-0.273	231\n'
#B += 'SM	0.37	0.33±0.061	0.12±0.005	0.008±0.004	-0.11±0.004	13511±13342	0.048	-3.159	18.964\n'
#B += 'DG	0.23±0	0.23±0	1.06±0	0.1±0.005	0.616±0.04	727694±41068	0.05	-2.6	11623'

<<<<<<< HEAD
# Gauss A
B  = 'ISM SI	1	2	0	-1.2	0.02\n'
B += 'ISM RI	0.4±0.49	1.6±0.478	0.0±0.0	-1.48±0.381	0.04±0.01\n'
B += 'SM SI	1	1.99	0	-1.791	0.404\n'
B += 'SM RI	1±0.0	1.79±0.398	0.0±0.0	-1.59±0.398	1.47±1.521\n'
B += 'DG SI	0	1.629	1	-1.316	1.732\n'
B += 'DG RI	0.93±0.196	0.99±0.0	0.03±0.104	-0.99±0.0	1.51±0.102\n'

# Gauss B
B = 'ISM SI	1	1.9	0	509	0.239\n'
B += 'ISM RI	0.8±0.4	1.9±0.011	0.0±0.0	605.53±192.918	0.18±0.038\n'
B += 'SM SI	1	1.9	0	509	0.413\n'
B += 'SM RI	0.4±0.49	1.9±0.031	0.6±0.49	1095.36±478.709	11.21±11.01\n'
B += 'DG SI	1	1.933	0	509.975	1595.174\n'
B += 'DG RI	0.1±0.32	0.61±0.17	0.9±0.32	101.5±30.8	1688±551'

# Moon
B = 'ISM SI	1	2	0	149.383	0.575\n'
B += 'ISM RI	0.4±0.409	2.0±0.001	0.5±0.496	269.98±42.885	3.07±1.357\n'
B += 'SM SI	1	1.998	0	149.383	2.653\n'
B += 'SM RI	0.27±0.335	2.0±0.0	0.49±0.488	276.77±18.122	258.47±373.734\n'
B += 'DG SI	1	1.998	0	149.697	359.188\n'
B += 'DG RI	0.09±0.267	1.27±0.553	0.8±0.398	161.65±101.831	3212.2±1368\n'

# Moon + N
B = 'ISM SI	1	2	0	15.3	0.3\n'
B += 'ISM RI	0.91±0.2	2.0±0	0.04±0.1	15.59±0.8	0.55±0.2\n'
B += 'SM SI	1	2	0	15.3	0.452\n'
B += 'SM RI	0.22±0.4	2.0±0	0.72±0.4	17.7±1	102±48.5\n'
B += 'DG SI	1	2	0	15.4	3836.3\n'
B += 'DG RI	0±0.0	1.6±0.04	1.0±0.0	16.2±0.06	3588±304\n'


## Flower
#B = 'ISM SI	0	0.41	0	20	0.061\n'
#B += 'ISM RI	0	0.41±0.0	0.0±0.0	19.51±0.0	0.05±0.012\n'
#B += 'SM SI	0	0.41	0	20	0.153\n'
#B += 'SM RI	0	0.3±0.179	0.01±0.012	0.72±0.483	0.81±0.596\n'
#B += 'DG SI	0	0.41	0	20	27.251\n'
#B += 'DG RI	0	0.35±0.019	0.2±0.385	20.63±1.361	37.03±5.432\n'


larger_smaller = [0,1,1,0,0,0]

def is_larger(val, rows, t, needLarger):
	if needLarger == 1:
		isLarger = True	
		for s in range(len(rows)):
			tmp_row = rows[s]
			tmp_row = tmp_row[t]
			tmp_row =  float(tmp_row.split(' ')[0])
			if val < tmp_row:
				isLarger = False
		return isLarger
=======

larger_smaller = [1,1,1,0,0,0,1,0,0]

lat = ''
rows = B.split('\n')
ISM = rows[0].split('\t')
SM = rows[1].split('\t')
DG = rows[2].split('\t')

	
print ISM
print SM
print DG

out_str = 'ISM'
for m in range(1, len(ISM)):
	ISM[m] = ISM[m].replace('±', ' $\pm$ ')
	SM[m] = SM[m].replace('±', ' $\pm$ ')
	DG[m] = DG[m].replace('±', ' $\pm$ ')

	SM1 =  float(SM[m].split(' ')[0])
	ISM1 = float(ISM[m].split(' ')[0])
	DG1 =  float(DG[m].split(' ')[0])


	if larger_smaller[m-1] == 1:
		if ISM1 >= SM1 and ISM1 >= DG1:
			out_str += '\n\t& \\textbf{' + ISM[m] + '} '
		else:
			out_str += '\n\t& ' + ISM[m] + ''
>>>>>>> 7fa38d169e52d53e5031bc3ebad54413b4e4c583
	else:
		isLarger = False
	
		for s in range(len(rows)):
			tmp_row = rows[s]
			tmp_row = tmp_row[t]
			tmp_row =  float(tmp_row.split(' ')[0])
			if val > tmp_row:
				isLarger = True

		return isLarger


lat = ''
rows = B.split('\n')

for r in range(len(rows)):
	test_row = rows[r].split('\t')
	if len(test_row) < 2:
		rows.remove(rows[r])
	else:
		rows[r] = rows[r].replace('±', ' $\pm$ ')
		rows[r] = rows[r].split('\t')


out_str = ''
for r in range(len(rows)):
	row = rows[r]

	for t in range(len(row)):
		if t == 0:
			out_str += '\n' + row[t]
		else:
			test_val =  float(row[t].split(' ')[0])

			if larger_smaller[t] == 1:
				if is_larger(test_val, rows, t, 1):
					out_str += '\n\t& \\textbf{' + row[t] + '} '
				else:
					out_str += '\n\t& ' + row[t] + ''
			else:
				if not is_larger(test_val, rows, t, 0):
					out_str += '\n\t& \\textbf{' + row[t] + '} '
				else:
					out_str += '\n\t& ' + row[t] + ''

	out_str += '\\\\'

print out_str




#out_str = 'ISM'

#
#	SM1 =  float(SM[m].split(' ')[0])
#	ISM1 = float(ISM[m].split(' ')[0])
#	DG1 =  float(DG[m].split(' ')[0])
#
#
#	if larger_smaller[m-1] == 1:
#		if ISM1 >= SM1 and ISM1 >= DG1:
#			out_str += '\n\t& \\textbf{' + ISM[m] + '} '
#		else:
#			out_str += '\n\t& ' + ISM[m] + ''
#	else:
#		if ISM1 <= SM1 and ISM1 <= DG1:
#			out_str += '\n\t& \\textbf{' + ISM[m] + '} '
#		else:
#			out_str += '\n\t& ' + ISM[m] + ''






#ISM = rows[0].split('\t')
#SM = rows[1].split('\t')
#DG = rows[2].split('\t')
#
#	
#print ISM
#print SM
#print DG
#
#out_str = 'ISM'
#for m in range(1, len(ISM)):
#	ISM[m] = ISM[m].replace('±', ' $\pm$ ')
#	SM[m] = SM[m].replace('±', ' $\pm$ ')
#	DG[m] = DG[m].replace('±', ' $\pm$ ')
#
#	SM1 =  float(SM[m].split(' ')[0])
#	ISM1 = float(ISM[m].split(' ')[0])
#	DG1 =  float(DG[m].split(' ')[0])
#
#
#	if larger_smaller[m-1] == 1:
#		if ISM1 >= SM1 and ISM1 >= DG1:
#			out_str += '\n\t& \\textbf{' + ISM[m] + '} '
#		else:
#			out_str += '\n\t& ' + ISM[m] + ''
#	else:
#		if ISM1 <= SM1 and ISM1 <= DG1:
#			out_str += '\n\t& \\textbf{' + ISM[m] + '} '
#		else:
#			out_str += '\n\t& ' + ISM[m] + ''
#
#
#out_str += '\\\\\n'
#
#out_str += SM[0]
#for m in range(1, len(SM)):
#	SM1 =  float(SM[m].split(' ')[0])
#	ISM1 = float(ISM[m].split(' ')[0])
#	DG1 =  float(DG[m].split(' ')[0])
#
#
#	if larger_smaller[m-1] == 1:
#		if SM1 >= ISM1 and SM1 >= DG1:
#			out_str += '\n\t& \\textbf{' + SM[m] + '} '
#		else:
#			out_str += '\n\t& ' + SM[m] + ''
#	else:
#		if SM1 <= ISM1 and SM1 <= DG1:
#			out_str += '\n\t& \\textbf{' + SM[m] + '} '
#		else:
#			out_str += '\n\t& ' + SM[m] + ''
#
#
#out_str += '\\\\\n'
#
#out_str += DG[0]
#for m in range(1, len(DG)):
#	SM1 =  float(SM[m].split(' ')[0])
#	ISM1 = float(ISM[m].split(' ')[0])
#	DG1 =  float(DG[m].split(' ')[0])
#
#
#	if larger_smaller[m-1] == 1:
#		if DG1 >= ISM1 and DG1 >= SM1:
#			out_str += '\n\t& \\textbf{' + DG[m] + '} '
#		else:
#			out_str += '\n\t& ' + DG[m] + ''
#	else:
#		if DG1 <= ISM1 and DG1 <= SM1:
#			out_str += '\n\t& \\textbf{' + DG[m] + '} '
#		else:
#			out_str += '\n\t& ' + DG[m] + ''
#
#
#
#out_str += '\\\\\n'
#
#
#
#
#
#print out_str
#
#
#
#
#
#
#
#
##for m in A.split('\t'):
##	V = m.split('±')
##	if len(V) == 1:
##		if bold:
##			lat += '\n&\\textbf{' + V[0] + '}'
##		else:
##			lat += '\n&' + V[0] 
##	else:
##		if bold:
##			lat += '\n&\\textbf{' + V[0] + ' $\pm$ ' + V[1] + '}'
##		else:
##			lat += '\n&' + V[0] + ' $\pm$ ' + V[1] + ''
##
##lat = lat + '\\\\'
##print lat
