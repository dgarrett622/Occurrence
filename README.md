# Occurrence
Collection of exoplanet occurrence rate data from [NASA's SAG 13](https://exoplanets.nasa.gov/exep/exopag/sag/#sag13) 
effort and Python scripts fitting occurrence rate models including stellar effective temperature to this data. Details 
of these models are contained in [Garrett et al. (2018)](https://doi.org/10.1088/1538-3873/aadff1). This work makes use of 
[`emcee`](https://github.com/dfm/emcee), a Python implementation of the affine-invariant ensemble sampler for Markov 
chain Monte Carlo (MCMC) proposed by [Goodman & Weare (2010)](http://dx.doi.org/10.2140/camcos.2010.5.65), written by
[Dan Foreman-Mackey](https://dfm.io/).

## Data
The data tables in the `Data` folder come from the [SAG 13 Google Drive](https://drive.google.com/drive/folders/0B520NCfkP4aOOW1SWDg2cHpYOVE)
folders `Burke`, `Mulders`, and `Natalie9p1`.

## MCMC Data
The data in these folders (stored as .npy files) are MCMC model parameter burn-in and sample data. The folder `MCMC Data All` contains 
parameter burn-in and samples for occurrence rate models fit to all of the data in the `Data` folder. The folder `MCMC 
Data FGK` contains parameter burn-in and samples for occurrence rate models fit to F, G, and K type star data contained 
in the `Data` folder. The folder `MCMC Data M` contains parameter burn-in and samples for occurrence rates fit only to M 
type star data contained in the `Data` folder.

## Scripts
The `Scripts` folder contains scripts which perform model fitting using MCMC sampling for "simple" and "break radius" 
models. Additional scripts are provided to display results of these fits.
