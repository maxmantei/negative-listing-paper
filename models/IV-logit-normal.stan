// Instrumental Variable Logit-Normal
// ----------------------------------
// 
// add short description
//
functions{
  vector min_max_logit(vector x){
    // function description
    vector[rows(x)] x_ast = logit((x - min(x))/(max(x) - min(x)));
    return(x_ast);
  }
  
  real max_abs_finite(vector x){
    // function describtion
    real new_max;
    real result;
    for(n in 1:rows(x)){
      if(is_inf(fabs(x[n])) == 1) continue;
      new_max = fmax(fabs(x[n]), result);
      result = new_max;
    }
    return(result);
  }
  
}

data {
  int<lower=0> N;
  int y_endogenous[N];       // endogenous variable
  vector[N] y;      // dependent variable
  int K;
  matrix[N,K] X;
  int M;          // number of other/instrumental variables
  matrix[N,M] Z;  // matrix of other/instrumental variables
  real threshold;
}

transformed data{
  // use the trick pointed out in the Stan manual
  int<lower=0> N_pos;
  int<lower=1,upper=N> n_pos[sum(y_endogenous)];
  int<lower=0> N_neg;
  int<lower=1,upper=N> n_neg[N - size(n_pos)];
  real max_y = max(y);
  real min_y = min(y);
  vector[N] y_ = min_max_logit(y);          // min-max normalization -> logit transformation
  real max_finite = max_abs_finite(y_);     // what is the maximum absolute finite transformed value?
  real maximum = fmax(threshold, max_finite); // use whatever is value is higher
  
  matrix[N,K+1] QX = qr_thin_Q(append_col(rep_vector(1.0,N),  X));
  matrix[K+1,K+1] RX = qr_thin_R(append_col(rep_vector(1.0,N),  X));
  matrix[N,K] QX_ast = (QX*RX[1,1])[,2:(K+1)];
  matrix[K,K] RX_ast = (RX/RX[1,1])[2:(K+1),2:(K+1)];
  matrix[K,K] RX_ast_inv = inverse(RX_ast);
  
  matrix[N,K+M+1] QXZ = qr_thin_Q(append_col(append_col(rep_vector(1.0,N), X), Z ));
  matrix[M+K+1,M+K+1] RXZ = qr_thin_R(append_col(append_col(rep_vector(1.0,N), X), Z ));
  matrix[N,K+M] QXZ_ast = (QXZ*RXZ[1,1])[,2:(K+M+1)];
  matrix[K+M,K+M] RXZ_ast = (RXZ/RXZ[1,1])[2:(K+M+1),2:(K+M+1)];
  matrix[K+M,K+M] RXZ_ast_inv = inverse(RXZ_ast);
  
  N_pos = size(n_pos);
  N_neg = size(n_neg);
  {
    int i;
    int j;
    i = 1;
    j = 1;
    for (n in 1:N) {
      if (y_endogenous[n] == 1) {
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
  real alpha_1;                 // intercept
  vector[K+M] params_tilde;     // coefficients
  vector<lower=0>[N_pos] z_pos; // latent variable for y == 1
  vector<upper=0>[N_neg] z_neg; // latent variable for y == 0
  real<lower=maximum> theta;
  real alpha_2;
  vector[K] beta_tilde;
  cholesky_factor_corr[2] L_Omega;
  real<lower=0> sigma;
  real zeta;
}

transformed parameters {
  matrix[2,2] L_Sigma = diag_pre_multiply([1, sigma]', L_Omega);
  vector[2] Y[N];
  vector[2] eta[N];
  
  for (n in 1:N_pos)
    Y[n_pos[n], 1] = z_pos[n];
  for (n in 1:N_neg)
    Y[n_neg[n], 1] = z_neg[n];
  for(n in 1:N){
    eta[n, 1] = alpha_1 + QXZ_ast[n,] * params_tilde;
    eta[n, 2] = alpha_2 + QX_ast[n,] * beta_tilde + Phi(Y[n, 1]) * zeta;
    if(y_[n] == positive_infinity()){
        Y[n, 2] = theta;
      } else if(y_[n] == negative_infinity()){
        Y[n, 2] = -theta;
      } else {
        Y[n, 2] = y_[n];
      }
  }
  
}

model {
  // likelihood
  target += multi_normal_cholesky_lpdf(Y | eta, L_Sigma);
  
  // Jacobian correction
  for(n in 1:N)
    target += log(max_y - min_y) + log_inv_logit(Y[n,2]) + log1m_inv_logit(Y[n,2]);

  // priors
  target += normal_lpdf(alpha_1 | 0, 2);
  target += normal_lpdf(alpha_2 | 0, 2);
  target += normal_lpdf(params_tilde | 0, 1.5);
  target += normal_lpdf(beta_tilde | 0, 1.5);
  target += normal_lpdf(sigma | 0, 2.5);
  target += lkj_corr_cholesky_lpdf(L_Omega | 3);
  target += normal_lpdf(zeta | 0, 1.5);
}

generated quantities{
  vector[K] gamma = (RXZ_ast_inv*params_tilde)[1:K];
  vector[M] delta = (RXZ_ast_inv*params_tilde)[(K+1):(K+M)];
  vector[K] beta = RX_ast_inv*beta_tilde;
  corr_matrix[2] Omega = multiply_lower_tri_self_transpose(L_Omega);
  //vector[N] y_rep;   // posterior replications of y
  //vector[N] log_lik; // pointwise posterior log-likelihood
  //for(n in 1:N){
  //  y_rep[n] = inv_logit(normal_rng(X[n,]*beta, sigma))*(max_y - min_y) + min_y;
  //  log_lik[n] = normal_lpdf(y_ast[n] | X[n,]*beta, sigma) + log(max_y - min_y) + 
  //               + log_inv_logit(y_ast[n]) + log1m_inv_logit(y_ast[n]);
  //}
}
