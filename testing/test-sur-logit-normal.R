

N <- 100
D <- 10

K <- 5

X <- matrix(rnorm(K*N), nrow = N, ncol = K)

sigma <- rexp(D, 1)

Omega <- diag(D)
for (i in 1:D){
  for(j in 1:D){
    if(i == j){ next}
    Omega[i,j] <- runif(1, -0.25, 0.25)
    Omega[j,i] <- Omega[i,j]
  }
}

errors <- MASS::mvrnorm(N, rep(0, D), diag(sigma) %*% Omega %*% diag(sigma))

sigma_beta <- rexp(K, 1) 
beta <- matrix(rnorm(K*D, 0, sd = sigma_beta), nrow = K, ncol = D)
alpha <- rnorm(D, 0, 1)

mu <- matrix(alpha, byrow = TRUE, nrow = N, ncol = D) + (X %*% beta)

Y_ast <- mu + errors

y_max <- 100
y_min <- 20

Y <- plogis(Y_ast)*(y_max - y_min) + y_min

for (d in 1:D){
  Y[which.max(Y[,d]),d] <- y_max
  Y[which.min(Y[,d]),d] <- y_min
}

standata <- list(
  N = N,
  D = D,
  K = K,
  X = X,
  y = Y,
  threshold = 5.9
)

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
#Sys.setenv(LOCAL_CPPFLAGS = '-march=native')


posterior <- stan(file = "models/sur-logit-normal.stan", data = standata, chains = 4, iter = 2000)

print(posterior, c("alpha", "beta"))

library(tidyverse)
library(broom)

tidyMCMC(posterior, "alpha", conf.int = TRUE, conf.level = 0.95) %>% 
  mutate(truth = alpha) %>% 
  ggplot(aes(x = term)) + 
    geom_point(aes(y = estimate), color = "red", alpha = 0.5) + 
    geom_point(aes(y = truth), color = "blue", alpha = 0.5) + 
    geom_linerange(aes(ymin=conf.low,ymax=conf.high), color = "red", alpha = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed") + 
    theme_minimal() +
    coord_flip()

tidyMCMC(posterior, "beta", conf.int = TRUE, conf.level = 0.95) %>% 
  mutate(truth = as.vector(beta)) %>% 
  separate(term, into = c("i", "j"), ",") %>% 
  mutate(i = str_extract(i, "\\d+") %>% as.numeric, j = str_extract(j, "\\d+") %>% as.numeric) %>% 
  ggplot(aes(x = i)) + 
  geom_point(aes(y = estimate), color = "red", alpha = 0.5) + 
  geom_point(aes(y = truth), color = "blue", alpha = 0.5) + 
  geom_linerange(aes(ymin=conf.low,ymax=conf.high), color = "red", alpha = 0.5) +
  facet_wrap(~j) + 
  geom_hline(yintercept = 0, linetype = "dashed") + 
  theme_minimal()

tidyMCMC(posterior, "sigma", conf.int = TRUE, conf.level = 0.95) %>% 
  mutate(truth = sigma) %>% 
  ggplot(aes(x = term)) + 
  geom_point(aes(y = estimate), color = "red", alpha = 0.5) + 
  geom_point(aes(y = truth), color = "blue", alpha = 0.5) + 
  geom_linerange(aes(ymin=conf.low,ymax=conf.high), color = "red", alpha = 0.5) +
  geom_hline(yintercept = 0) + 
  theme_minimal() +
  coord_flip()

tidy(posterior, "Omega", conf.int = TRUE, conf.level = 0.95) %>% 
  mutate(truth = as.vector(Omega)) %>% 
  separate(term, into = c("i", "j"), ",") %>% 
  mutate(i = str_extract(i, "\\d+") %>% as.numeric, j = str_extract(j, "\\d+") %>% as.numeric) %>% 
  ggplot(aes(x = i)) + geom_point(aes(y = estimate), color = "red", alpha = 0.5) + 
  geom_linerange(aes(ymin=conf.low, ymax=conf.high), color = "red", alpha = 0.5) + 
  geom_point(aes(y = truth), color = "blue", alpha = 0.5) + 
  facet_wrap(~j) + 
  geom_hline(yintercept = 0, linetype = "dashed") + 
  theme_minimal()



