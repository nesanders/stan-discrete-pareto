import pickle, os
import numpy as np
import pystan
import mpmath
import scipy
from scipy.stats._distn_infrastructure import rv_discrete
from matplotlib import pyplot as plt

## Load color list
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']



##############################
## Implementation of the discrete power law in scipy
##############################

class gzipf_gen(rv_discrete):
    r"""A Zipf discrete random variable. with variable y_min
    %(before_notes)s
    Notes
    -----
    The probability mass function for `zipf` is:
    .. math::
        f(k, a) = \frac{1}{\zeta(a,q) k^a}
    for :math:`k \ge 1`.
    `zipf` takes :math:`a` as shape parameter. :math:`\zeta` is the
    zeta function (`scipy.special.zeta`) with offset :math:`q`
    %(after_notes)s
    %(example)s
    """
    def _argcheck(self, alpha, q):
        return alpha > 0
        return q > 0

    def _pmf(self, k, alpha, q):
        Pk = 1.0 / scipy.special.zeta(alpha, q) / k**(alpha)
        return Pk

def zipf_dist(alpha, y_min): 
    gzipf = gzipf_gen(a=y_min, b=10000, name='gen zipf', longname='Generalized Zipf')
    return gzipf(alpha=alpha, q=y_min)


##############################
## Stan Definitions
##############################

## Compile sampling model
with open('discrete_power_law_varying_Ymin.stan', 'r') as f:
    stan_model = f.read()

sm = pystan.StanModel(model_code=stan_model, 
        includes=[os.getcwd()+'/hurwitz_zeta.hpp'], 
        allow_undefined=True) #, verbose=True) 

## Compile test functions
with open('hurwitz_testing.stan', 'r') as f:
    stan_test_func = f.read()

sm_test = pystan.StanModel(model_code = stan_test_func, 
        includes=[os.getcwd()+'/hurwitz_functions.hpp'], 
        allow_undefined=True) #, verbose=True) 


##############################
## Inference test
##############################
test_alpha = 1.5 # True alpha value to test at
test_ymins = [1, 10, 100, 1000] #ymin values to test
test_N = 1000 # Sample size of test dataset

# Sample from discrete Pareto using scipy
sim_y = {}; y_vals = {}; stan_data = {}
fit = {}; ext = {}
for i_tx, test_ymin in enumerate(test_ymins):
    sim_y[test_ymin] = zipf_dist(alpha=test_alpha, y_min=test_ymin).rvs(test_N)
    # Identify value range
    y_vals[test_ymin] = np.arange(min(sim_y[test_ymin]), max(sim_y[test_ymin])+1)
    # Create Stan input dic
    stan_data[test_ymin] = {
        'K': len(y_vals[test_ymin]),
        'values': y_vals[test_ymin],
        'frequencies': [sum(sim_y[test_ymin]==y) for y in y_vals[test_ymin]],
        'alpha_rate': 1,
        'alpha_shape': 0.5,
        'y_min': test_ymin,
        }

    ## Fit the model
    fit[test_ymin] = sm.sampling(data = stan_data[test_ymin], iter=500, chains=8)
    ext[test_ymin] = fit[test_ymin].extract()

## Plot inference comparison
plt.figure()
for i_tx, test_ymin in enumerate(test_ymins):
    plt.hist(ext[test_ymin]['alpha'], bins=25, 
             label='$y_{\\rm{min}}='+str(test_ymin)+'$', histtype='step', lw=2)
plt.axvline(test_alpha, color='k', ls='dashed', label='True value')
plt.xlabel('$\\alpha$')
plt.ylabel('$p$')
plt.legend(title='True $y_{\\rm{min}}$')
plt.savefig('test_pareto_discrete_inference.png', dpi=300)


##############################
## Hurwitz zeta test
##############################

## Compare function calls to reference values
test_alphas = np.linspace(1.5, 100)
scipy_vals = {
    'zeta':np.zeros([len(test_ymins), len(test_alphas)]),
    'zeta_prime':np.zeros([len(test_ymins), len(test_alphas)]),
    }
mpmath_vals = {
    'zeta':np.zeros([len(test_ymins), len(test_alphas)]),
    'zeta_prime':np.zeros([len(test_ymins), len(test_alphas)]),
    }
stan_vals = {
    'zeta':np.zeros([len(test_ymins), len(test_alphas)]),
    'zeta_prime':np.zeros([len(test_ymins), len(test_alphas)]),
    }
## Calculate zeta and zeta prime for variety of y_min and alpha values
for i_tym, test_ymin in enumerate(test_ymins):
    for i_ta, test_alpha in enumerate(test_alphas):
        scipy_vals['zeta'][i_tym, i_ta] = scipy.special.zeta(test_alpha, test_ymin)
        mpmath_vals['zeta'][i_tym, i_ta] = mpmath.zeta(test_alpha, test_ymin)
        mpmath_vals['zeta_prime'][i_tym, i_ta] = mpmath.zeta(test_alpha, test_ymin, 1)
    sm_out = sm_test.sampling(
            data={'N':len(test_alphas), 'y_min':test_ymin, 'alpha':test_alphas}, 
            iter=1, algorithm='Fixed_param', chains=1)
    stan_vals['zeta'][i_tym] = sm_out.extract()['zeta']
    stan_vals['zeta_prime'][i_tym] = sm_out.extract()['zeta_prime']

## Plot comparison of reference functions
fig, axs = plt.subplots(2, 2, sharex='all')
for i_tym, test_ymin in enumerate(test_ymins):
    axs[0,0].plot(test_alphas, (stan_vals['zeta'][i_tym] - mpmath_vals['zeta'][i_tym]).T, label=test_ymin, color=colors[i_tym])
    axs[1,0].plot(test_alphas, (mpmath_vals['zeta'][i_tym] - scipy_vals['zeta'][i_tym]).T, label=test_ymin, color=colors[i_tym])
    axs[0,1].plot(test_alphas, (stan_vals['zeta_prime'][i_tym] - mpmath_vals['zeta_prime'][i_tym]).T, label=test_ymin, color=colors[i_tym])

axs[0,0].legend(title='True $y_{\\rm{min}}$')
plt.xlabel('$\\alpha$')
#for ax in axs: ax.set_yscale('symlog')
plt.suptitle('Numerical calculation difference')
axs[0,0].set_ylabel('stan - mpmath')
axs[1,0].set_ylabel('mpmath - scipy')
axs[1,1].set_visible(False)

axs[0,0].set_title('zeta')
axs[0,1].set_title('zeta_prime')

plt.savefig('test_pareto_discrete_comparefuncs.png', dpi=300)


## Plot the Hurwitz zeta itself
plt.figure()
for i_tym, test_ymin in enumerate(test_ymins):
    plt.plot(test_alphas, mpmath_vals['zeta'][i_tym], label=test_ymin)
plt.legend()
plt.xlabel('$\\alpha$')
plt.ylabel('Hurwitz zeta (mpmath)')
plt.axvline(1, ls='dashed', color='0.5')
plt.savefig('test_hurwitz_zeta.png', dpi=300)
plt.semilogy()
plt.savefig('test_hurwitz_zeta_log.png', dpi=300)
