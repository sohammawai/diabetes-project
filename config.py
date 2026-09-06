# config.py
import os
N_WORKERS = 4
THREADS_PER_WORKER = 4
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)