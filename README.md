libspmv
-------

Proof of concept MatMult method for distributed vector using Eigen library 
SparseMatrix templates

* initially used Eigen::SparseMatrix
* performance improvement with MKL
* Uses a Local to global map "L2GMap" to update ghost vectors
* Can read in PETSc binary matrix and vector formats

