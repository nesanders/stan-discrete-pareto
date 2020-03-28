library(poweRlaw)
library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = 4)

true.xmin <- 200 
true.alpha <- 2.2

n <- 2E5

X <- rpldis(n = n, xmin = true.xmin, alpha = true.alpha)

compress_data <- function(x){
  raw <- table(x)
  vv <- as.numeric(names(raw))
  ns <- as.numeric(raw)
  return(list(v = vv, fs = ns, K = length(vv)))
}

cdata <- compress_data(X)

plaw.data <- list(
  K = cdata$K,
  values = cdata$v,
  frequencies = cdata$fs,
  y_min = true.xmin,
  alpha_shape = 1, ## placing a Gamma prior over alpha. Could also be a Jeffrey's prior (leads to proper posterior)
  alpha_rate = 1
)

plaw <- stan_model("discrete_power_law_varying_Ymin.stan",
                   allow_undefined = TRUE,
                   verbose = TRUE, ## change this to TRUE during development, it'll help figure out why things don't work
                   includes = paste0('\n#include "',
                                     file.path(getwd(), 'hurwitz_zeta.hpp'), '"\n') 
)                   

opt <- optimizing(plaw, data = plaw.data)
mcmc <- sampling(plaw, data = plaw.data)
 
alpha.samples <- extract(mcmc, 'alpha')$alpha

hist(alpha.samples, probability = TRUE, xlab = expression(alpha))
abline(v = true.alpha, lwd = 2)
abline(v = opt$par, lwd = 2, lty = 2)
legend(x = "topright", legend = c("true", "MAP"), lty = 1:2, bty = 'n')