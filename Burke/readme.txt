% Occurrence rate analysis follows Burke et al. 2015, ApJ, 809,8 (B15)
% Based upon Q1-Q16 Kepler Planet candidate catalog (Mullally et

% al. 2015, ApJS, 217, 31) and pipeline completeness model (Christiansen

% et al. 2015, ApJ, 810, 95).  
% GK sample given first for 4200 K < Teff < 6100 K
% M sample given second for Teff < 4200 K
% Changes from B15:
% 
1) Expanded region of fitting.  Results come from two separate fits

% over the 10<Porb<50 day and 50<Porb<300 day.  The short period region

% was fit over 0.35<Rp<12.0 Rearth and the long period region was fit

% over 0.75<Rp<12.0 Rearth.


% 2) More complicated parametric model.  To accomodate the expanded

% region of fitting from B15, the parametric model has a constant term

% added to the Rp broken powerlaw and the Porb power law exponent has a

% dependence on Rp (P^(beta + beta1*log(rp) + beta2*log(rp)^2)).
% 3) Includes planet radius uncertainties.  The likelihood is modified 
% in order to account for the planet radius 
% uncertainties from the MCMC KOI fitting of Rowe et al. 2015, ApJS, 217, 16.
% ***Warnings the uncertainties are statistical error bars only.  They do NOT 
% include systematic effects that are larger than the statistical error bar (see B15).