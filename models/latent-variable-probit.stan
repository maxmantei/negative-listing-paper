// Latent Variable Probit
// ----------------------
// 
// add short description
//
data {
  int<lower=0> N; // Number of observations
  int y[N];       // dependent variable
  int K;          // number of control variables
  matrix[N,K] X;  // matrix of control variables
  int M;          // number of other/instrumental variables
  matrix[N,M] Z;  // matrix of other/instrumental variables
}

transformed data{
  // use the trick pointed out in the Stan manual
  int<lower=0> N_pos;
  int<lower=1,upper=N> n_pos[sum(y)];
  int<lower=0> N_neg;
  int<lower=1,upper=N> n_neg[N - size(n_pos)];
  N_pos = size(n_pos);
  N_neg = size(n_neg);
  {
    int i;
    int j;
    i = 1;
    j = 1;
    for (n in 1:N) {
      if (y[n] == 1) {
        n_pos[i] = n;
        i += 1;
      } else {
        n_neg[j] = n;
        j += 1;
      }
    }
  }
}

parameters {
  vector[K] gamma;              // coefficients for X
  vector[M] delta;              // coefficients for Z
  vector<lower=0>[N_pos] z_pos; // latent variable for y == 1
  vector<upper=0>[N_neg] z_neg; // latent variable for y == 0
}

transformed parameters {
  // putting together the latent variable vector
  vector[N] z;
  for (n in 1:N_pos)
    z[n_pos[n]] = z_pos[n];
  for (n in 1:N_neg)
    z[n_neg[n]] = z_neg[n];
}

model {
  // likelihood of the latent variable (sigma fixed to 1 -> Probit)
  target += normal_lpdf(z | X*gamma + Z*delta, 1);

  // priors on regression coefficients
  target += normal_lpdf(gamma | 0, 1.5);
  target += normal_lpdf(delta | 0, 1.5);
}

generated quantities {
  vector[N] eta = Phi(X*gamma + Z*delta); // linear predictor
  int y_rep[N];                           // posterior replication
  vector[N] log_lik;                      // pointwise log-likelihood
  
  for (n in 1:N){
    y_rep[n] = bernoulli_rng(eta[n]);
    log_lik[n] = bernoulli_lpmf(y[n] | eta[n]);
  }
}
