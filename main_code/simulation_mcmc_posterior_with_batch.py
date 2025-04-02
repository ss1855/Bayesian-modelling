from INDISCIM import seird_model
from scipy.integrate import odeint
import numpy as np
import h5py
from multiprocessing import Pool
import psutil
from INDISCIM import seird_model
from scipy.integrate import odeint
import numpy as np
from para_comp_initialization import fixed_params
from multiprocessing import Pool


def simulate_single_sample(sample_index, posterior_samples, t_full, intervals, initial_conditions):
    """
    Simulates the SEIRD model for a single posterior sample index across all intervals.
    """
    current_initial_conditions = initial_conditions
    simulation_set = []  # Store simulations for this posterior sample index

    for interval_index, (start, end) in enumerate(intervals):
        t_interval = t_full[start:end + 1]

        # Get the posterior parameters for the current interval and sample index
        params = posterior_samples[interval_index][sample_index, :]

        # Simulate the SEIRD model for the current interval
        simulation = odeint(
            seird_model,
            current_initial_conditions,
            t_interval,
            args=(*params, *fixed_params)
        )
        simulation_set.append(simulation)

        # Update initial conditions for the next interval
        current_initial_conditions = simulation[-1, :]

    # Combine all interval simulations for this sample index
    combined_simulation = simulation_set[0]
    for sim in simulation_set[1:]:
        combined_simulation = np.vstack((combined_simulation[:-1], sim))  # Stack, removing overlap

    return combined_simulation


def monitor_memory():
    """
    Prints the current memory usage of the script.
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")


def simulate_with_posterior_samples_in_batches(posterior_samples, t_full, intervals, initial_conditions,
                                               batch_size=10000, n_processes=4, output_file="simulations.h5"):
    """
    Simulates the SEIRD model using posterior samples in batches and saves results incrementally.

    Parameters:
        posterior_samples (list of np.ndarray): Posterior samples for each interval.
        t_full (np.ndarray): Full time vector for the simulation.
        intervals (list of tuple): List of time intervals (start, end) for each simulation.
        initial_conditions (list or np.ndarray): Initial conditions for the SEIRD model.
        batch_size (int): Number of samples to process in each batch.
        n_processes (int): Number of processes to use for parallelization.
        output_file (str): Path to the output file for storing results.

    Returns:
        None. Results are saved incrementally to `output_file`.
    """
    n_samples = posterior_samples[0].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size  # Compute the number of batches

    # Prepare HDF5 file for saving results
    with h5py.File(output_file, "w") as f:
        # Run a single simulation to determine result shape
        example_result = simulate_single_sample(0, posterior_samples, t_full, intervals, initial_conditions)
        result_shape = example_result.shape

        # Create a dataset for storing results
        dset = f.create_dataset(
            "simulations",
            shape=(n_samples, *result_shape),
            dtype="float32",  # Use float32 for lower memory usage
            chunks=(batch_size, *result_shape),
            compression="gzip"  # Compress data to save space
        )

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            # Prepare arguments for the batch
            args = [
                (sample_index, posterior_samples, t_full, intervals, initial_conditions)
                for sample_index in range(start_idx, end_idx)
            ]

            # Run simulations in parallel
            with Pool(processes=n_processes) as pool:
                batch_results = pool.starmap(simulate_single_sample, args)

            # Save batch results to HDF5 file
            dset[start_idx:end_idx, :, :] = np.array(batch_results, dtype="float32")
            print(f"Batch {batch_idx + 1}/{n_batches} saved. Memory status:")
            monitor_memory()


# Example usage
