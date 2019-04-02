// Seemingly Unrelated Logit-Normal
// --------------------------------
// 
// add short description
//
functions{
  vector min_max_logit(vector x){
    // function description
    vector[rows(x)] x_ast = logit((x - min(x))/(max(x) - min(x)));
    return(x_ast);
  }
  
  real array_min(vector[] x){
    // function describtion
    int dim1 = dims(x)[1];
    int dim2 = dims(x)[2];
    real new_min;
    real result;
    for(d in 1:dim1){
      for(n in 1:dim2){
        new_min = fmin(fabs(x[d,n]), result);
        result = new_min;
      }
      return(result);
    }
  }
  
  real array_max(vector[] x){
    // function describtion
    int dim1 = dims(x)[1];
    int dim2 = dims(x)[2];
    real new_min;
    real result;
    for(d in 1:dim1){
      for(n in 1:dim2){
        new_min = fmax(fabs(x[d,n]), result);
        result = new_min;
      }
      return(result);
    }
  }
  
  real max_abs_finite(vector[] x){
    // function describtion
    int dim1 = dims(x)[1];
    int dim2 = dims(x)[2];
    real new_max;
    real result;
    for(d in 1:dim1){
      for(n in 1:dim2){
        if(is_inf(fabs(x[d,n])) == 1) continue;
        new_max = fmax(fabs(x[d,n]), result);
        result = new_max;
      }
      return(result);
    }
  }
  
}

data {
  int<lower=0> N;
  int<lower=1> D;
  vector[D] y[N];
  int K;
  matrix[N,K] X;
  real threshold;
}

transformed data{
  real max_y = array_max(y);
  real min_y = array_min(y);
  vector[D] y_[N];
  real max_finite;     
  real maximum;
  matrix[N,K+1] Q = qr_thin_Q(append_col(rep_vector(1.0,N),  X));
  matrix[K+1,K+1] R = qr_thin_R(append_col(rep_vector(1.0,N),  X));
  matrix[N,K] Q_ast = (Q*R[1,1])[,2:(K+1)];
  matrix[K,K] R_ast = (R/R[1,1])[2:(K+1),2:(K+1)];
  matrix[K,K] R_ast_inv = inverse(R_ast);
  
  for (d in 1:D)
    for(n in 1:N)
      y_[n,d] = logit((y[n,d] - 20.0)/(100.0 - 20.0));
  
  max_finite = max_abs_finite(y_); // what is the maximum absolute finite transformed value?
  maximum = fmax(threshold, max_finite); // use whatever is value is higher
  
  print("threshold: ", threshold, " | max finite: ", max_finite, " | chosen max: ", maximum)
}

parameters {
  real<lower=maximum> theta;
  vector[D] alpha;
  real<lower=0> sigma_alpha;
  matrix[K,D] beta_tilde;
  vector<lower=0>[K] sigma_beta_tilde;
  vector<lower=0>[D] sigma;
  cholesky_factor_corr[D] L_Omega;
}

transformed parameters {

}

model {
  // expectation
  vector[D] mu[N];
  vector[D] y_ast[N];
  for(d in 1:D){
    for(n in 1:N){
      if(y_[n,d] == positive_infinity()){
          y_ast[n,d] = theta;
        } else if(y_[n,d] == negative_infinity()){
          y_ast[n,d] = -theta;
        } else {
          y_ast[n,d] = y_[n,d];
        }
    }
  }
  
  for (n in 1:N)
    mu[n] = alpha + (Q_ast[n,]*beta_tilde)';

  
  // likelihood
  target += multi_normal_cholesky_lpdf(y_ast | mu, diag_pre_multiply(sigma, L_Omega));
  
  // Jacobian correction
  for (n in 1:N)
    target += log(100 - 20) + log_inv_logit(y_ast[n]) + log1m_inv_logit(y_ast[n]);
  
  // priors
  target += normal_lpdf(alpha | 0, sigma_alpha);
  target += exponential_lpdf(sigma_alpha | 1);
  for (k in 1:K)
    target += normal_lpdf(beta_tilde[k,] | 0, sigma_beta_tilde[k]);
  target += exponential_lpdf(sigma_beta_tilde | 1);
  target += exponential_lpdf(sigma | 1);
  target += lkj_corr_cholesky_lpdf(L_Omega | 4);
}

generated quantities{
  corr_matrix[D] Omega = multiply_lower_tri_self_transpose(L_Omega);
  matrix[K,D] beta = R_ast_inv*beta_tilde;
  //vector[N] y_rep;   // posterior replications of y
  //vector[N] log_lik; // pointwise posterior log-likelihood
  //for(n in 1:N){
  //  y_rep[n] = inv_logit(normal_rng(alpha + X[n,]*beta, sigma))*(max_y - min_y) + min_y;
  //  log_lik[n] = normal_lpdf(y_ast[n] | alpha + X[n,]*beta, sigma) + log(max_y - min_y) + 
  //               + log_inv_logit(y_ast[n]) + log1m_inv_logit(y_ast[n]);
  //}
}
