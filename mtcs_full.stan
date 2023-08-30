functions {
	// likelihood for each nu_i (measured signal)
	real nu_likelihood_lpdf(row_vector nu_i, row_vector theta_i, real alpha_0, matrix phi) {
		real norm_t;
		real density;
		norm_t = dot_self(nu_i - theta_i*phi');
		density = pow(2*pi()/alpha_0, size(nu_i)/2.0)*exp((-alpha_0/2.0)*norm_t);
		return log(density);
	}

	// hierachical prior for each theta_i (spase solution)
	real theta_multprior_lpdf(row_vector theta_i, vector alpha) {
		real log_sum = 0;
		for (j in 1:size(alpha)) {
			log_sum +=  normal_lpdf(theta_i[j] | 0, alpha[j]);
		}
		return log_sum;
	}

	// prior for alpha
	real alpha_prior_lpdf(vector alpha) {
		real log_sum = 0;
		real c = 1;
		real d = 1;
		for (j in 1:size(alpha)) {
			log_sum +=  gamma_lpdf(alpha[j] | c, d);
		}
		return log_sum;
	}

}

data {
	// number of signals
	int M;
	// original size of the signal
	int n;
	// measured size of the signal
	int m;
	// data: matrix representing all signals
	matrix[M, m] X; // measured signal: vector[m] nu_i;
	// measure matrix
	matrix[m, n] phi;
	// dictionary
	// matrix[n, n] D;

}

parameters {
	// matrix representing all thetas
	matrix[M, n] Theta;
	// hyperparameter alpha
	vector<lower=0, upper=20>[n] alpha;
	// hyperparameter alpha_0
	real<lower=0> alpha_0;
}

/* transformed parameters {
	real real_signal[M][n];
	for (j in 1:M) {
		real_signal[j] = Theta[j] * transpose(D)
	}
} */

model {
	alpha_0 ~ normal(0, 1);
	alpha ~ alpha_prior();
	// theta_i ~ theta_like(alpha)
	for (j in 1:M) {
		Theta[j] ~ theta_multprior(alpha);
	}
	// nu_i ~ nu_like(theta_i, alpha_0, phi)
	for (j in 1:M) {
		X[j] ~ nu_likelihood(Theta[j], alpha_0, phi);
	}	
}
