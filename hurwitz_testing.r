library(rstan)

rstan_options(auto_write = TRUE)



funs <- stan_model("hurwitz_testing.stan",
                   allow_undefined = TRUE,
                   verbose = FALSE, ## change this to TRUE during development, it'll help figure out why things don't work
                   includes = paste0('\n#include "',
                                     file.path(getwd(), 'hurwitz_functions.hpp'), '"\n') 
)                   
##
improvised_hurwitz <- function(x) VGAM::lerch(x = .9999999999999999, s = x, v = compute.data$y_min)
improvised_hurwitz <- Vectorize(improvised_hurwitz)
improvised_hurwitz_prime <- function(x) numDeriv::grad(improvised_hurwitz, x)
###
true.xmin <- 125
true.alpha <- 2.2

compute.data <- list(
  y_min = true.xmin,
  alpha = true.alpha
)

res <- sampling(funs, data = compute.data, iter = 1, algorithm="Fixed_param", chains = 1, refresh = 0)
extract(res, 'zeta')
extract(res, 'zeta_prime')

if(true.xmin == 1){
  cat("zeta:", VGAM::zeta(compute.data$alpha), " zeta_prime:", VGAM::zeta(compute.data$alpha, deriv = 1), "\n")
}


improvised_hurwitz(compute.data$alpha)
improvised_hurwitz_prime(compute.data$alpha)


