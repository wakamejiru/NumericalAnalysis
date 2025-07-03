import numpy as np
import matplotlib.pyplot as plt

# rngをグローバルに定義
rng = np.random.default_rng()

def simulate_langevin(step_max, dt):
    """
    1D Langevin equation single simulation.
    Cv is returned as v[i]*v[0].
    """
    t: np.ndarray = np.zeros(int(step_max + 1))
    x: np.ndarray = np.zeros(int(step_max + 1))
    v: np.ndarray = np.zeros(int(step_max + 1))
    Cv: np.ndarray = np.zeros(int(step_max + 1))
    MSD: np.ndarray = np.zeros(int(step_max + 1))

    x[0] = 0.0
    v[0] = rng.standard_normal() # Sample initial velocity from standard normal distribution
    
    # Cv[0] = v[0]*v[0]
    Cv[0] = v[0] * v[0] 

    for i in range(1, int(step_max + 1)):
        t[i] = i * dt
        x[i] = x[i-1] + dt * v[i-1]
        v[i] = v[i-1] - dt * v[i-1] + np.sqrt(2.0 * dt) * rng.standard_normal()
        
        # Calculate Cv as v[i]*v[0]
        Cv[i] = (v[i] * v[0])
        MSD[i] = np.power(x[i] - x[0], 2)
    
    return Cv, MSD

def plot_results(t, avg_Cv, Cv_th, avg_MSD, MSD_th, num_samples):
    """
    Plots averaged Cv and MSD against their theoretical solutions and saves them as PNG.
    """
    plt.figure(figsize=(14, 7)) # Adjust figure size

    # Plot for Velocity Autocorrelation Function (Cv)
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st subplot
    plt.plot(t[1:], avg_Cv[1:], label=f'Simulated Avg Cv (N={num_samples})', color='blue', alpha=0.7)
    plt.plot(t[1:], Cv_th[1:], label='Theoretical Cv', color='red', linestyle='--')
    plt.xscale('log') # Set x-axis to log scale
    plt.xlabel('Time (log scale)')
    plt.ylabel('Velocity Autocorrelation Function')
    plt.title('Velocity Autocorrelation Function')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7') # Add grid lines

    # Plot for Mean Squared Displacement (MSD)
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd subplot
    plt.plot(t[1:], avg_MSD[1:], label=f'Simulated Avg MSD (N={num_samples})', color='green', alpha=0.7)
    plt.plot(t[1:], MSD_th[1:], label='Theoretical MSD', color='purple', linestyle='--')
    plt.xscale('log') # Set x-axis to log scale
    plt.yscale('log') # Set y-axis to log scale (MSD often looks better on log scale)
    plt.xlabel('Time (log scale)')
    plt.ylabel('Mean Squared Displacement (log scale)')
    plt.title('Mean Squared Displacement')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7') # Add grid lines

    plt.tight_layout() # Adjust spacing between subplots automatically
    
    # Save the figure as a PNG file
    filename = f'brownian_motion_N{num_samples}.png'
    plt.savefig(filename, dpi=300) # Save with 300 DPI for better quality
    plt.close() # Close the figure to free up memory

# --- Main execution part ---
if __name__ == "__main__":
    STEP_MAX: int = 10000
    DT: float = 1.0e-2

    # Calculate theoretical solutions (only once)
    t_theoretical = np.arange(0, (STEP_MAX + 1) * DT, DT)
    
    # Theoretical Cv: <v(t)v(0)> = <v(0)^2> * exp(-t)
    # Since v[0] follows standard normal distribution, <v(0)^2> = 1.
    Cv_th = np.exp(-t_theoretical) 
    
    # Theoretical MSD: <x(t)^2>
    MSD_th = 2.0 * (t_theoretical - (1.0 - np.exp(-t_theoretical)))

    # Run simulations with different number of samples
    sample_counts = [1, 10, 100, 1000, 10000, 100000] # Vary number of samples to confirm

    for num_sample in sample_counts:
        print(f"Running simulation with {num_sample} samples...")
        
        # Initialize arrays to accumulate results
        Cv_sum: np.ndarray = np.zeros(int(STEP_MAX + 1))
        MSD_sum: np.ndarray = np.zeros(int(STEP_MAX + 1))

        for _ in range(num_sample):
            Cv_tmp, MSD_tmp = simulate_langevin(STEP_MAX, DT)
            Cv_sum += Cv_tmp
            MSD_sum += MSD_tmp

        # Calculate averages
        Cv_avg = Cv_sum / float(num_sample)
        MSD_avg = MSD_sum / float(num_sample)

        # Plot and save results for the current number of samples
        plot_results(t_theoretical, Cv_avg, Cv_th, MSD_avg, MSD_th, num_sample)
        print(f"Plot saved as brownian_motion_N{num_sample}.png.")