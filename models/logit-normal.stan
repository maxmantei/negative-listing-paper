// Logit-Normal with infinite values
// ---------------------------------
// 
// add short description
//
functions{
  vector min_max_logit(vector x){
    // function description
    vector[rows(x)] x_ast = inv_logit((x - min(x))/(max(x) - min(x)));
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
  
  vector solve_inv_logit(vector x, vector treshold, real[] x_r, int[] x_i) {
    vector[1] deltas;
    deltas = inv_logit(x) - treshold;
    return deltas;
  }
}

data {
  int<lower=0> N;
  vector[N] y;
  int K;
  matrix[N,K] X;
  vector[1] threshold;
}

transformed data{
  real max_y = max(y);
  real min_y = min(y);
  vector[N] y_ = min_max_logit(y);          // min-max normalization -> logit transformation
  real max_finite = max_abs_finite(y_);  // what is the maximum absolute finite transformed value?
  real x_r[0];                              // only needed for the algebra slover
  int x_i[0];                               // only needed for the algebra slover
  // The following line seeks the user-specified threshold 
  // for maximum values and calculates the respective logit
  vector[1] max_threshold = algebra_solver(solve_inv_logit, [5]', threshold, x_r, x_i);
  real maximum = fmax(max_threshold[1], max_finite); // use whatever is value is higher
}

parameters {
  real<lower=maximum> theta; 
  vector[K] beta;
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
  y_ast ~ normal(X*beta, sigma);
  
  // Jacobian correction
  target += log(max_y - min_y) + log_inv_logit(y_ast) + log1m_inv_logit(y_ast);
  
  // priors
  beta ~ normal(0, 1.5);
  sigma ~ exponential(1);
}

generated quantities{
  vector[N] y_rep;   // posterior replications of y
  vector[N] log_lik; // pointwise posterior log-likelihood
  for(n in 1:N){
    y_rep[n] = inv_logit(normal_rng(X[n,]*beta, sigma))*(max_y - min_y) + min_y;
    log_lik[n] = 0; // need to derive this ... probably impossible, because infinite values... ?!
  }
}
