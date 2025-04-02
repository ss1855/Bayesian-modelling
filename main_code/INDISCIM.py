# SEIRD Model Core (Module 1)
# ===========================

# Import necessary libraries
import numpy as np
from scipy.integrate import odeint
from para_comp_initialization import fixed_params

# SEIRD Model Definition
def seird_model(y, t, beta, gamma_e, lambda_a, lambda_p, lambda_m, lambda_s, rho, delta, eff_a, eff_p, eff_m, eff_s, mu, alpha, total_population):
    S, E, I_a, I_p, I_m, I_s, H, R, D, C = y

    new_infections = (beta * S * (eff_a * I_a + eff_p * I_p + eff_m * I_m + eff_s * I_s)) / total_population

    dSdt = -new_infections
    dEdt = new_infections - gamma_e * E
    dIadt = alpha * gamma_e * E - lambda_a * I_a
    dIpdt = (1 - alpha) * gamma_e * E - lambda_p * I_p
    dImdt = mu * lambda_p * I_p - lambda_m * I_m
    dIsdt = (1 - mu) * lambda_p * I_p - lambda_s * I_s
    dHdt = lambda_s * I_s - rho * H
    dRdt = lambda_a * I_a + lambda_m * I_m + (1 - delta) * rho * H
    dDdt = delta * rho * H
    dCdt = (1 - alpha) * gamma_e * E 
    return [dSdt, dEdt, dIadt, dIpdt, dImdt, dIsdt, dHdt, dRdt, dDdt, dCdt]

# Solve SEIRD Model
def solve_seird(params, initial_conditions, t_interval):
    beta, gamma_e, lambda_a, lambda_p, lambda_m, lambda_s, rho, delta = params
    y = odeint(seird_model, initial_conditions, t_interval, args=(beta, gamma_e, lambda_a, lambda_p, lambda_m, lambda_s, rho, delta, *fixed_params))
    return y.T

# INDISCIM.py

# Parameter Bounds
PARAM_BOUNDS = {
    "beta": (0.005, 1),
    "gamma_e": (0.4, 0.6),
    "lambda_a": (0.1, 0.2),
    "lambda_p": (0.1, 0.2),
    "lambda_m": (0.1, 0.2),
    "lambda_s": (0.12, 0.22),
    "rho": (0.01, 0.1),
    "delta": (0.1, 0.5),
}
