functions{
    real hurwitz_zeta(real s, real a);
}
data{
  int<lower=0> K; // number of unique values
  int values[K]; // y-values
  real<lower=0> y_min; // minimum y-value
  int<lower=0> frequencies[K]; // number of counts at each y-value
  real<lower=0> alpha_shape; // gamma hyperprior parameters for alpha
  real<lower=0> alpha_rate;
}
parameters{
  real <lower=1> alpha;
}
model{
  real constant = log(hurwitz_zeta(alpha, y_min));
  for (k in 1:K) {
    target += frequencies[k] * (-alpha * log(values[k]) - constant);
  }
  target += gamma_lpdf(alpha |alpha_shape, alpha_rate);
} 
/*TODO: implement rng for prior and posterior predictive checks*/
