functions{
    real hurwitz_zeta(real s, real a);
    real hurwitz_zeta_prime(real s, real a);
}
data{
    int<lower=1> N;
    real<lower=0> y_min; 
    real<lower=1> alpha[N];
}
generated quantities{
    real zeta[N] ;
    real zeta_prime[N];
    
    for (n in 1:N) {
        zeta[n] = hurwitz_zeta(alpha[n], y_min);
        zeta_prime[n] = hurwitz_zeta_prime(alpha[n], y_min);
    }
}
