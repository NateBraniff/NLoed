# function for power analysis with given design and model

#use sigma points (or if needed monte carlo simulation) to determine 'power curve', shrinking of parameter error (bias +var) with n design replications
#perhaps combine with rounding methods?? to allow for rounded variations of the same design.
# rounding should probably happen here

#include a prediction variance plot for verifying genral equivlance theorem plot (either as a var vs X or a stem plot for support points in high dim)
#FDS-like plots, prediction variance limit vs fraction of design space with lower variance than limit, used to compare designs and sample sizes, along with CI's (ellipses or intervals)
#d-efficiency plots for comparing sample size (normed to approx max for each sample size) vs relative efficiency (normed to lowest sample size approx or rounded exact)
#    shows 'real'improvment from extra samples rather than distance from unachievable ideal, howevver regular d-efficiency may motivate adding a single point to achieve near theoretical max