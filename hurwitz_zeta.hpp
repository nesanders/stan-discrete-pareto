#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
using boost::math::isfinite;

inline double hurwitz_zeta(const double& s, const double& xmin, std::ostream* pstream__){
  boost::math::quadrature::exp_sinh<double> integrator;
  auto int_rep_hurwitz_zeta = [&](double t) {
    double ans =  (std::pow(t, s-1) * std::exp(-xmin*t))/(1-std::exp(-t));
    if(!isfinite(ans)) return 0.0; // weird patching to get quadrature working
    return ans;
  };
  double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
  double error = 0.0;
  double L1 = 0.0;
  size_t levels;
  double zeta = integrator.integrate(int_rep_hurwitz_zeta, tolerance, &error, &L1, &levels)/boost::math::tgamma(s);
  return zeta;
}

inline var hurwitz_zeta(const var& s_var, const double& a, std::ostream* pstream__) {
  double s = s_var.val();
  double f = hurwitz_zeta(s, a, pstream__);
  boost::math::quadrature::exp_sinh<double> integrator;
  auto int_rep_hurwitz_zeta_prime = [&](double x) {
    double ans = std::exp((1-a)*x) * std::pow(x, s-1) *( std::log(x) - boost::math::digamma(s))/(std::exp(x) -1);
    if(!isfinite(ans)) return 0.0; // weird patching to get quadrature working
    return ans;
  };
  double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
  double error = 0.0;
  double L1 = 0.0;
  size_t levels;
  double deriv = integrator.integrate(int_rep_hurwitz_zeta_prime, tolerance, &error, &L1, &levels)/boost::math::tgamma(s);
  return var(new precomp_v_vari(f, s_var.vi_, deriv));
}
