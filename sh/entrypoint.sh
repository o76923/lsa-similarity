#!/bin/sh

#PYTHON_UNBUFFERED=1 mpiexec -n 1 python -m py.delegator
#PYTHON_UNBUFFERED=1 mpiexec -n 1 python -m cProfile -o /app/data/lsa.profile /app/py/delegator.py
PYTHON_UNBUFFERED=1 python -m py.delegator
