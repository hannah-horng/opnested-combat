# opnested-combat
This is the repository for OPNested, OPNested+GMM, and OPNested-ComBat from the manuscript entitled ["Improved generalized ComBat methods for harmonization of radiomic features" by Horng et al.](https://rdcu.be/cZf2n), published at Nature Scientific Reports. These methods are updates to the algorithms previously published at [generalized-combat](https://github.com/hannah-horng/generalized-combat). All functions needed to implement the methods are in the OPNestedComBat.py file.

## OPNestedComBat
Enables harmonization by multiple imaging parameters. See file comments for more details.

## OPNested+GMM ComBat
Uses Gaussian Mixture Modeling (GMM) to identify scan groupings associated with hidden covariates to better address bimodal feature distributions. OPNested+GMM treats the GMM grouping as a batch variable for harmonization.

## OPNested-GMM ComBat
Uses Gaussian Mixture Modeling (GMM) to identify scan groupings associated with hidden covariates to better address bimodal feature distributions. OPNested-GMM treats the GMM grouping as a clinical covariate for protection during harmonization.

# Example
To illustrate how to use the function file, an example with CAPTK features extracted from the publicly available NSCLC-Radiogenomics dataset has been added.

# Updates
These methods are still a work in progress! Keep an eye out for future updates to the methods. 
