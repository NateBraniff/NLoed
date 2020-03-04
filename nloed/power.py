# function for power analysis with given design and model

#pre N: asymptotic confidence intervals only 
#enter an N range Nmin to Nmax (or just single N)
#discretize with multiple N -> sigma/boostrap CI's/Cov's at each N (jitter at each N?!) [graphical, thershold]
#                           -> FDS/D-eff(Opt-eff) plots [graphical, thershold]
#                           -> what about T optimality designs, what about bayesian CI's??

# Discretize (depends on N):    basic Adam's apportionment, 
#                               other rounding options (naive...)
# Sample Size Selection (N?):   assesment of confidence interval size with design replication or N selection
#                               similar FDS/D-eff/relative D-eff plots for selecting sample size
# Fine Tuning:                  jitter/tuning of exact design



#use sigma points (or if needed monte carlo simulation) to determine 'power curve', shrinking of parameter error (bias +var) with n design replications
#perhaps combine with rounding methods?? to allow for rounded variations of the same design.
# rounding should probably happen here

#include a prediction variance plot for verifying genral equivlance theorem plot (either as a var vs X or a stem plot for support points in high dim)
#FDS-like plots, prediction variance limit vs fraction of design space with lower variance than limit, used to compare designs and sample sizes, along with CI's (ellipses or intervals)
#d-efficiency plots for comparing sample size (normed to approx max for each sample size) vs relative efficiency (normed to lowest sample size approx or rounded exact)
#    shows 'real'improvment from extra samples rather than distance from unachievable ideal, howevver regular d-efficiency may motivate adding a single point to achieve near theoretical max