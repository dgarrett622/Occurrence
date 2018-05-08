Planet Catalog: Q1-Q16 SOC 9.1
- adjusted the star and planet properties using the DR24 star properties catalog which contained singificantly more contributions from confirmed planet papers.
- also adjusted any new confirmations of small planets in the habitable zone using literature values.
- fixed a few errors with anomalous Rp/R* and b values from light curve modeling
- fixed a few anomalous M star properties in the DR24 catalog

Filters: 
- Removed candidates with Bootstrap False Alarm Rate > 1e-12
- removed stars with logg < 4.0

Compeleteness Model:  
- Christiansen et al. 2015 ApJ 810 95
- used the analytic approximation to the window function as described by Burke et al 2015

Vetting Efficiency: 
- TCERT detection efficiency unavailable for SOC 9.1
- Consequently, assume 100% vetting efficiency

Reliability: 
- Astrophysical FPPs unavailable
- Instrumental FPP's not quantified excepting via bootstrap statistic
- Consequently, effectively assume 100% reliability after application of bootstrap FAR as a filter

Methodology:  
- Non-parametric
- Direct Calculation (i.e. Inverse Detection Efficiency) 
- statistical uncertainties only
- NaN's assigned to bins with zero planet candidates



