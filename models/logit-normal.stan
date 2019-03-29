// Logit-Normal with infinite values
// ---------------------------------
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
  vector[N] y;
  int K;
  matrix[N,K] X;
  real threshold;
}

transformed data{
  real max_y = max(y);
  real min_y = min(y);
  vector[N] y_ = min_max_logit(y);          // min-max normalization -> logit transformation
  real max_finite = max_abs_finite(y_);     // what is the maximum absolute finite transformed value?
  real maximum = fmax(threshold, max_finite); // use whatever is value is higher
  matrix[N,K+1] Q = qr_thin_Q(append_col(rep_vector(1.0,N),  X));
  matrix[K+1,K+1] R = qr_thin_R(append_col(rep_vector(1.0,N),  X));
  matrix[N,K] Q_ast = (Q*R[1,1])[,2:(K+1)];
  matrix[K,K] R_ast = (R/R[1,1])[2:(K+1),2:(K+1)];
  matrix[K,K] R_ast_inv = inverse(R_ast);
}

parameters {
  real<lower=maximum> theta;
  real alpha;
  vector[K] beta_tilde;
  real<lower=0> sigma;
}

transformed parameters {
  vector[N] y_ast;
  for(n in 1:N){
    if(y_[n] == positive_infinity()){
        y_ast[n] = theta;
      } else if(y_[n] == negative_infinity()){
        y_ast[n] = -theta;
      } else {
        y_ast[n] = y_[n];
      }
  }
}

model {
  // likelihood
  target += normal_lpdf(y_ast | alpha +  Q_ast*beta_tilde, sigma);
  
  // Jacobian correction
  target += log(max_y - min_y) + log_inv_logit(y_ast) + log1m_inv_logit(y_ast);
  
  // priors
  target += normal_lpdf(alpha | 0, 2.5);
  target += normal_lpdf(beta_tilde | 0, 2.5);
  target += exponential_lpdf(sigma | 1);
}

generated quantities{
  vector[K] beta = R_ast_inv*beta_tilde;
  vector[N] y_rep;   // posterior replications of y
  vector[N] log_lik; // pointwise posterior log-likelihood
  for(n in 1:N){
    y_rep[n] = inv_logit(normal_rng(alpha + X[n,]*beta, sigma))*(max_y - min_y) + min_y;
    log_lik[n] = normal_lpdf(y_ast[n] | alpha + X[n,]*beta, sigma) + log(max_y - min_y) + 
                 + log_inv_logit(y_ast[n]) + log1m_inv_logit(y_ast[n]);
  }
}
