# Copula Negative Log-Likelihood Functions

## Overview

This repository provides Python and R implementations for computing the negative log-likelihood of several widely-used Copula families: **Clayton, Frank, Gaussian (GS), Gumbel, and Student's t**. 

These functions are designed to help estimate copula parameters via maximum likelihood. They are particularly valuable for multivariate dependency modeling in geospatial data analysis. For example, fitting joint distributions using these scripts is a standard methodological step when integrating disparate climate variables (like precipitation and Snow Water Equivalent or soil moisture) to compute standardized agricultural and hydrological drought indices.

## Repository Structure

* `negloglike.py`: Contains the complete suite of copula negative log-likelihood functions vectorized for Python (`numpy`/`scipy`).
* `negloglike_clayton.R`: R implementation for the Clayton copula.
* `negloglike_frank.R`: R implementation for the Frank copula.
* `negloglike_gs.R`: R implementation for the Gaussian copula.
* `negloglike_gumbel.R`: R implementation for the Gumbel copula.
* `negloglike_t.R`: R implementation for the Student's t copula.

## Usage & Integration

These functions expect matrices of uniform margins $U \in (0,1)^d$ as input `u` (typically obtained via empirical CDFs or parametric marginal distributions). 

* **R Users**: Source the individual files as needed or compile them into an internal package. Ensure inputs are matrices or dataframes where `rowSums` and `colSums` can be safely applied.
* **Python Users**: Import the necessary functions from `negloglike.py`. Ensure inputs are `numpy.ndarray` objects to leverage the vectorized `axis=1` operations.

---

