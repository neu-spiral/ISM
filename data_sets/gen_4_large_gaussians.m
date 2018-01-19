#!/usr/bin/octave


%--------------------
N_of_samples = 2^8; 					% number of samples
N_of_dimensions = 2^2;
%--------------------


for N_of_samples=6:8
	for N_of_dimensions=6:15
		NSample = 2^N_of_samples;
		NDims = 2^N_of_dimensions;

		D = 2;							%	number of data dimensions
		p = 10;
		n = NSample/4;				%	number of samples per cluster
		noise_d = NDims - 2;
		
		n_samples = n*4;
		
		a  =  randn(n,D)  +  repmat([0,p],n,1);
		b  =  randn(n,D)  +  repmat([p,0],n,1);
		c  =  randn(n,D)  +  repmat([0,-p],n,1);
		d  =  randn(n,D)  +  repmat([-p,0],n,1);
		
		dat  =  [a;b;c;d];
		noise = 60*rand(n_samples,noise_d) - 30;
		
		X = [dat, noise];
		X = X - repmat(mean(X),  n_samples,  1);
		X = round(X*100)/100;
		
		
		%plot(X(:,1), X(:,2), 'x');
		%hold on
		%plot(X(:,3), X(:,4), 'x');
		
		%save KDAC_test.mat X
		outFname = ['Dimond_gaussians_', num2str(NSample), '_', num2str(NDims), '.csv']
		csvwrite(outFname,X, 'precision', 3)


	end
end


