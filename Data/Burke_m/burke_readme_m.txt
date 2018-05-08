Occurrence rate analysis follows Burke et al. 2015, ApJ, 809,8 (B15)
http://adsabs.harvard.edu/abs/2015ApJ...809....8B

Based upon Q1-Q16 Kepler Planet candidate catalog (Mullally et
al. 2015, ApJS, 217, 31) and pipeline completeness model (Christiansen
et al. 2015, ApJ, 810, 95).  The files eta_m.txt, sigma_p_m.txt, and
sigma_n_m.txt are for the combined M dwarf sample selected according
to the criteria outlined in B15 and lower Teff than the K dwarf selection.

Changes from B15:

1) Expanded region of fitting.  Results come from two separate fits
over the 10<Porb<50 day and 50<Porb<175 day.  The short period region
was fit over 0.4<Rp<10.0 Rearth and the long period region was fit
over 0.85<Rp<10.0 Rearth.

2) More complicated parametric model.  To accomodate the expanded
region of fitting from B15, the parametric model has a constant term
added to the Rp broken powerlaw and the Porb power law exponent has a
dependence on Rp (P^(beta + beta1*log(rp) + beta2*log(rp)^2)).  For
the short period M dwarfs beta2=0.0 was fixed for the fit, and for the
long period M dwarfs beta1=0.0 and beta2=0.0 was fixed for the fit.

3) Includes planet radius uncertainties.  The likelihood is modified in order to account for the planet radius uncertainties from the MCMC KOI fitting of Rowe et al. 2015, ApJS, 217, 16.

***Warnings the uncertainties are statistical error bars only.  They do NOT include systematic effects that are larger than the statistical error bar (see B15).***

Includes a file extrap_m.txt which is 1/0 flag with 1 indicating the grid value includes extrapolating the parameteric model.
