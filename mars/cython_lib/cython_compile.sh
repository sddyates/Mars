export CFLAGS=-I/Users/sdaley/Work/Programs/miniconda3/pkgs/numpy-base-1.17.2-py37h6575580_0/lib/python3.7/site-packages/numpy/core/include/

cythonize -3 -i --annotate *.pyx
