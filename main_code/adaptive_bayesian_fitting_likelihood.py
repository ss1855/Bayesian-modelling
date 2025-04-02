from INDISCIM import seird_model,PARAM_BOUNDS
from scipy.integrate import odeint
import numpy as np
import emcee
from para_comp_initialization import fixed_params
from multiprocessing import Pool
from emcee.backends import HDFBackend
def log_likelihood(params, initial_conditions, t_interval, observed_ci_interval, observed_cd_interval,sigma_ME=0.0001, sigma=0.01,):
    # Simulate the SEIRD model
    y = odeint(seird_model, initial_conditions, t_interval, args=(*params,*fixed_params))
    _, _, _, _, _, _, _, _, D, C = y.T

    # Calculate total variance
    sigma_ci = np.sqrt(sigma_ME**2 + sigma**2)
    sigma_cd = np.sqrt(sigma_ME**2 + sigma**2)

    # Gaussian log-likelihood for infections
    ci_error = -0.5 * np.sum(((np.log(C + 1e-6) - np.log(observed_ci_interval + 1e-6)) / sigma_ci) ** 2 + np.log(2 * np.pi * sigma_ci**2))

    # Gaussian log-likelihood for deaths
    cd_error = -0.5 * np.sum(((np.log(D + 1e-6) - np.log(observed_cd_interval + 1e-6)) / sigma_cd) ** 2 + np.log(2 * np.pi * sigma_cd**2))

    return ci_error + cd_error

# Define the log-prior function

def log_prior(params):
    #nonlocal rejected_count
    for param_name, value in zip(PARAM_BOUNDS.keys(), params):
        lower, upper = PARAM_BOUNDS[param_name]  # Access bounds directly using the parameter name
        if not (lower <= value <= upper):
            #print(f"Parameter {param_name}={value} is out of bounds ({lower}, {upper})")
            #rejected_count += 1
            #print(f"Total rejected proposals: {rejected_count}")
            return -np.inf  # Reject parameter if out of bounds
    return 0.0  # All parameters are within bounds


# Define the log-posterior function
def log_posterior(params, initial_conditions, t_interval, observed_ci_interval, observed_cd_interval):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, initial_conditions, t_interval, observed_ci_interval, observed_cd_interval)

def adaptive_bayesian_fitting_weekly_paper_likelihood(observed_ci, observed_cd, t_full, intervals, initial_conditions, current_guess,  n_walkers=50, n_steps=1000, burn_in=200):
    """
    Adaptive Bayesian fitting with fixed σ_ME and σ for weekly SEIRD model fitting.

    Parameters:
        observed_ci (np.ndarray): Observed cumulative infections.
        observed_cd (np.ndarray): Observed cumulative deaths.
        t_full (np.ndarray): Full time vector.
        intervals (list of tuple): List of start and end indices for each interval.
        initial_conditions (list): Initial state variables for the SEIRD model.
        current_guess (list): Initial guess for model parameters.
        sigma_ME (float): Fixed measurement error variance.
        sigma (float): Fixed intrinsic scatter variance.
        n_walkers (int): Number of walkers for MCMC sampling.
        n_steps (int): Total number of MCMC steps.
        burn_in (int): Number of burn-in steps to discard.

    Returns:
        list: Fitted parameters (posterior means) for each interval.
        list: Simulated results for each interval.
        list: Posterior samples after burn-in for each interval.
    """
    all_fitted_params = []
    all_results = []
    all_posterior_samples = []  # To store posterior samples for each interval
    #rejected_count = 0
    # Define the log-likelihood function

# MCMC for each interval
    for start, end in intervals:
        if end + 1 > len(t_full):
            end = len(t_full) - 1

        t_interval = t_full[start:end+1]
        observed_ci_interval = observed_ci[start:end+1]
        observed_cd_interval = observed_cd[start:end+1]

        # Debugging shapes
        #print(f"start: {start}, end: {end}")
        #print(f"t_interval length: {len(t_interval)}, observed_ci_interval length: {len(observed_ci_interval)}, observed_cd_interval length: {len(observed_cd_interval)}")

    # Ensure lengths match
        if len(t_interval) != len(observed_ci_interval) or len(t_interval) != len(observed_cd_interval):
            raise ValueError(f"Length mismatch: t_interval={len(t_interval)}, observed_ci_interval={len(observed_ci_interval)}, observed_cd_interval={len(observed_cd_interval)}")

        ndim = len(current_guess)
        initial_positions = [
           [np.random.uniform(*PARAM_BOUNDS[param]) for param in PARAM_BOUNDS.keys()]
           for _ in range(n_walkers)
        ]
        print(initial_conditions)

        # Use multiprocessing for walkers
        with Pool(processes=4)  as pool:
            backend = HDFBackend(f"mcmc_interval_{start}_{end}.h5")
            backend.reset(n_walkers, ndim)
            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, log_posterior,
                args=(initial_conditions, t_interval, observed_ci_interval, observed_cd_interval),
                pool=pool,
                backend = backend
            )
            sampler.run_mcmc(initial_positions, n_steps, progress=True)


        posterior_samples = sampler.get_chain(discard=burn_in, flat=True)
        all_posterior_samples.append(posterior_samples)
        ###
        import matplotlib.pyplot as plt

        # After sampler.run_mcmc
        chains = sampler.get_chain()  # Shape: (n_steps, n_walkers, ndim)

        # Plot trace plots for each parameter
        for i, param_name in enumerate(PARAM_BOUNDS.keys()):
            plt.figure(figsize=(10, 6))
            for walker in range(n_walkers):
                plt.plot(chains[:, walker, i], alpha=0.5, label=f"Walker {walker}" if walker < 10 else None)  # Only label a few walkers for clarity
            plt.title(f"Trace Plot for Parameter: {param_name}")
            plt.xlabel("Step")
            plt.ylabel(param_name)
            plt.tight_layout()
            plt.savefig(f"trace_plot_{param_name}_interval_{start}_{end}.png")  # Save plots for each interval
            plt.close()
      ###


        fitted_params = np.mean(posterior_samples, axis=0)
        all_fitted_params.append(fitted_params)

        y_final = odeint(seird_model, initial_conditions, t_interval, args=(*fitted_params, *fixed_params))
        all_results.append(y_final)

        initial_conditions = y_final[-1, :]
        current_guess = fitted_params

    return all_fitted_params, all_results, all_posterior_samples
