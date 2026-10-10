# Source before launching sweeps: one BLAS/OpenMP thread per process. LSODA's LAPACK calls on the
# dense Hessian Jacobian otherwise start OpenBLAS threads in every worker (load average ~175 seen).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
