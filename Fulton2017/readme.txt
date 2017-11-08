Planet Catalog: Q1-Q16 Mullally et al. (2015)


Stellar Catalog for planet detections: California-Kepler Survey (Petigura et al. (2017))

Stellar Catalog for Kepler target stars: Stellar17 (Mathur et al. (2016))


Disclaimer:
Fulton et al. (2017) carefully analyzed candidates orbiting stars with 4700 < Teff < 6500 K,
P < 100 days, and Rp > 1.14 Re. The robustness of occurrence rates calculated outside that
regime is uncertain. Uncertainties have not been adjusted using the simulations described 
in the appendix of Fulton et al. (2017). They are likely underestimated by a factor of 2-3.


Filters: 
- Removed candidates with m_i < 6 (see eqn. 2 of Fulton et al. (2017))
- Removed false positives (from Paper I of CKS series, Petigura et al. (2017))
- Removed candidates orbiting hosts with Kp > 14.2
- Removed candidates with b > 0.7
- Removed candidates with P > 640 d (differs from Fulton et al. (2017))
- Removed giant host stars: R*[Rs] > 10^(0.00025*(Teff[K] - 5500)+0.20) (eqn. 1 of Fulton et al. (2017))
- Removed host stars with Teff > 6000 K or Teff < 5000 K (G stars only, differs from Fulton et al. (2017))


Compeleteness Model:  
- followed prescription in Christiansen et al. (2016) using the SOC 9.1(?) injections from 
   Christiansen et al. (2015)
- fit a Gamma CDF function with k=12.7, l=1.00 (fixed), theta=0.71
   (see eqn. 3 of Fulton et al. (2017) coeffs differ from Fulton et al. (2017) due to 
   different stellar sample


Vetting Efficiency: 
- Assume 100% vetting efficiency


Reliability: 
- Assume 100% reliability (after false positive filter)


Methodology:  
- See Fulton et al. (2017)
- Non-parametric
- Direct Calculation (i.e. Inverse Detection Efficiency) 
- statistical uncertainties only
- NaN's assigned to bins with zero planet candidates


Indicies:
	(i0, j0, k0)  =  (1, 0, 3)
	
	