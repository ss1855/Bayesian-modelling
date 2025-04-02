# ===========================
# Interval Management (Module 3)
# ===========================
from scipy.optimize import minimize
from scipy.integrate import odeint  # Needed for solving SEIRD model during fitting
import numpy as np                 # For numerical operations

def generate_intervals_with_overlap(n, interval_length=7):
    intervals = []
    start = 0
    while start < n:
        end = min(start + interval_length - 1, n)
        intervals.append((start, end))
        start = end
    return intervals
