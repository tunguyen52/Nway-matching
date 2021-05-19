# Nway Cross Matching

## Introduction
This Github repository contains the Python implementation of the two X-match algorithms, CanILP and DirILP, as presented in the paper "Globally optimal and scalable N-way matching of astronomy catalogs". In addition, we provided a sample application of DirILP on a dataset provided by the Hyper Suprime-Cam (HSC) Subaru Strategic Survey. The full data can be found at https://lsst.ncsa.illinois.edu/~yusra/nway_test/pdr1_cosmos/HSC-I/ 

In particular, the notebook [DirILP Test - Special Case](DirILP%20Test%20-%20Special%20Case.ipynb) contains an implementation of DirILP's Special case, where the astronometric uncertainty of each detection is the same, as outlined in Appendix A and its performance on a simulated dataset. The more general version of DirILP, together with CanILP, is implemented in [CanILP + DirILP General case](CanILP%20+%20DirILP%20General%20case.ipynb). Finally, we showed how parallel processing can be utilized to speed up the cross-matching procedure in [DirILP Special Case - Application.py](DirILP%20Special%20Case%20-%20Application.py). To run the file, you need to extract [data.7z](data.7z) to get objs.pkl, which is a portion of the data from HSC PDR1 survey. 

## Required Packages:
* Numpy: 1.17.2
* Pandas: 0.25.1
* Scipy: 1.3.1
* Scikit-learn: 0.21.3
* Matplotlib: 3.1.3
* Gurobi Optimizer version 9.0.1
