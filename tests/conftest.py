"""
Pytest configuration for the quant-platform test suite.

Force single-threaded BLAS/OpenBLAS so background compute threads are not
still live when Python finalises after the test session.  Without this,
numpy's OpenBLAS worker threads can call std::terminate() during cleanup,
causing a SIGABRT (exit 134) on Linux CI runners even though all tests pass.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
