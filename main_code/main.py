# ===========================
# Main Script (Module 6)
# ===========================
from adaptive_bayesian_fitting import adaptive_bayesian_fitting_weekly
from INDISCIM import seird_model, solve_seird
from para_comp_initialization import initialize_parameters, initialize_conditions
from intervals import generate_intervals_with_overlap
from adaptive_fitting import adaptive_fitting_weekly
from simulation import simulate_with_multiple_params
from simulation_mcmc_posterior import simulate_with_posterior_samples
from adaptive_bayesian_fitting import adaptive_bayesian_fitting_weekly
from adaptive_bayesian_fitting_likelihood import adaptive_bayesian_fitting_weekly_paper_likelihood
from para_comp_initialization import fixed_params
from simulation_mcmc_posterior_with_batch import simulate_with_posterior_samples_in_batches
import numpy as np
import os
import pandas as pd
import time
if __name__ == "__main__":

    start_time = time.time()

    n = 100
    t = np.linspace(0, n, n + 1)
    intervals = generate_intervals_with_overlap(n)
    print(intervals)

    params = initialize_parameters()
    initial_conditions = initialize_conditions()
    current_guess = list(params.values())
    file_path = os.path.join("C:/Users/Sujan Shrestha/Desktop/my_code/input_directory", "India_COVID_rolling_average.xlsx")
    data = pd.ExcelFile(file_path)


    # Load the data from the first sheet
    df = data.parse('Sheet1')

    # Define n (number of rows to extract)

    # Get the first n+1 rows for 'Cinfected' and 'Cdeath'
    subset = df[['Cinfected', 'Cdeath']].iloc[:n+1]

    # Store 'Cinfected' and 'Cdeath' in two separate arrays
    observed_ci = subset['Cinfected'].to_numpy()
    observed_cd = subset['Cdeath'].to_numpy()



    #fitted_params, results = adaptive_fitting_weekly(observed_ci, observed_cd, t, intervals, initial_conditions, current_guess)
    fitted_params, results,posterior_samples = adaptive_bayesian_fitting_weekly_paper_likelihood(observed_ci, observed_cd, t, intervals, initial_conditions, current_guess)
    print(f"Posterior sample set length: {len(posterior_samples)}")
    print(f"Intervals length: {len(intervals)}")
    for j, posterior_sample_set in enumerate(posterior_samples):
        print(f"Posterior sample set {j + 1} shape: {posterior_sample_set.shape}")
        print(posterior_sample_set[:5])
    #print(f"Accessing interval index: {i}")
    #total_simulation = simulate_with_multiple_params(fitted_params, t, intervals, initial_conditions)

    #combined_simulations = simulate_with_posterior_samples(posterior_samples, t, intervals, initial_conditions)
    #array = np.array(combined_simulations)
    #print(array.shape)

    combined_simulations =simulate_with_posterior_samples_in_batches(
        posterior_samples=posterior_samples,
        t_full=t,
        intervals=intervals,
        initial_conditions=initial_conditions,
        batch_size=10000,  # Process 10,000 samples at a time
        n_processes=4,  # Number of parallel processes
        output_file="seird_simulations.h5"  # Output file for saving results
    )


# Save the array
    output_dir = "output_directory"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    #os.makedirs(output_dir, exist_ok=True)  # Create directory if not exists
    file_path = os.path.join(output_dir, f"posterior_samples_interval.npy")
    np.save(file_path, posterior_samples)
    file_path = os.path.join(output_dir, f"fitted_samples_interval.npy")
    np.save(file_path, fitted_params)
    file_path = os.path.join(output_dir, "combined_simulations1000.npy")

    np.save(file_path, combined_simulations)
    print(f"Array saved to {file_path}")
    #print(combined_simulations.shape)
    #print("Simulation completed. Total results shape:", total_simulation.shape)
    end_time = time.time()  # End timer
    print(f"Total script execution time: {end_time - start_time:.2f} seconds.")
