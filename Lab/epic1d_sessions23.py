#!/usr/bin/env python3
#
# Electrostatic PIC code in a 1D cyclic domain
import numpy as np
from numpy import arange, concatenate, zeros, linspace, floor, array, pi
from numpy import sin, cos, sqrt, random, histogram, abs, sqrt, max
from scipy.signal import find_peaks
import matplotlib.pyplot as plt # Matplotlib plotting library
import time
import os


try:
    import matplotlib.gridspec as gridspec  # For plot layout grid
    got_gridspec = True
except:
    got_gridspec = False

# Need an FFT routine, either from SciPy or NumPy
try:
    from scipy.fftpack import fft, ifft
except:
    # No SciPy FFT routine. Import NumPy routine instead
    from numpy.fft import fft, ifft

def rk4step(f, y0, dt, args=()):
    """ Takes a single step using RK4 method """
    k1 = f(y0, *args)
    k2 = f(y0 + 0.5*dt*k1, *args)
    k3 = f(y0 + 0.5*dt*k2, *args)
    k4 = f(y0 + dt*k3, *args)

    return y0 + (k1 + 2.*k2 + 2.*k3 + k4)*dt / 6.

def calc_density(position, ncells, L):
    """ Calculate charge density given particle positions
    
    Input
      position  - Array of positions, one for each particle
                  assumed to be between 0 and L
      ncells    - Number of cells
      L         - Length of the domain

    Output
      density   - contains 1 if evenly distributed
    """
    # This is a crude method and could be made more efficient
    
    density = zeros([ncells])
    nparticles = len(position)
    
    dx = L / ncells       # Uniform cell spacing
    for p in position / dx:    # Loop over all the particles, converting position into a cell number
        plower = int(p)        # Cell to the left (rounding down)
        offset = p - plower    # Offset from the left
        density[plower] += 1. - offset
        density[(plower + 1) % ncells] += offset
    # nparticles now distributed amongst ncells
    density *= float(ncells) / float(nparticles)  # Make average density equal to 1
    return density

def periodic_interp(y, x):
    """
    Linear interpolation of a periodic array y at index x
    
    Input

    y - Array of values to be interpolated
    x - Index where result required. Can be an array of values
    
    Output
    
    y[x] with non-integer x
    """
    ny = len(y)
    if len(x) > 1:
        y = array(y) # Make sure it's a NumPy array for array indexing
    xl = floor(x).astype(int) # Left index
    dx = x - xl
    xl = ((xl % ny) + ny) % ny  # Ensures between 0 and ny-1 inclusive
    return y[xl]*(1. - dx) + y[(xl+1)%ny]*dx

def fft_integrate(y):
    """ Integrate a periodic function using FFTs
    """
    n = len(y) # Get the length of y
    
    f = fft(y) # Take FFT
    # Result is in standard layout with positive frequencies first then negative
    # n even: [ f(0), f(1), ... f(n/2), f(1-n/2) ... f(-1) ]
    # n odd:  [ f(0), f(1), ... f((n-1)/2), f(-(n-1)/2) ... f(-1) ]
    
    if n % 2 == 0: # If an even number of points
        k = concatenate( (arange(0, n/2+1), arange(1-n/2, 0)) )
    else:
        k = concatenate( (arange(0, (n-1)/2+1), arange( -(n-1)/2, 0)) )
    k = 2.*pi*k/n
    
    # Modify frequencies by dividing by ik
    f[1:] /= (1j * k[1:]) 
    f[0] = 0. # Set the arbitrary zero-frequency term to zero
    
    return ifft(f).real # Reverse Fourier Transform
   

def pic(f, ncells, L):
    """ f contains the position and velocity of all particles
    """
    nparticles = len(f) // 2     # Two values for each particle
    pos = f[0:nparticles] # Position of each particle
    vel = f[nparticles:]      # Velocity of each particle

    dx = L / float(ncells)    # Cell spacing

    # Ensure that pos is between 0 and L
    pos = ((pos % L) + L) % L
    
    # Calculate number density, normalised so 1 when uniform
    density = calc_density(pos, ncells, L)
    
    # Subtract ion density to get total charge density
    rho = density - 1.
    
    # Calculate electric field
    E = -fft_integrate(rho)*dx
    
    # Interpolate E field at particle locations
    accel = -periodic_interp(E, pos/dx)

    # Put back into a single array
    return concatenate( (vel, accel) )

####################################################################

def run(pos, vel, L, ncells=None, out=[], output_times=linspace(0,20,100), cfl=0.5):
    
    if ncells == None:
        ncells = int(sqrt(len(pos))) # A sensible default

    dx = L / float(ncells)
    
    f = concatenate( (pos, vel) )   # Starting state
    nparticles = len(pos)
    
    time = 0.0
    for tnext in output_times:
        # Advance to tnext
        stepping = True
        while stepping:
            # Maximum distance a particle can move is one cell
            dt = cfl * dx / max(abs(vel))
            if time + dt >= tnext:
                # Next time will hit or exceed required output time
                stepping = False
                dt = tnext - time
            f = rk4step(pic, f, dt, args=(ncells, L))
            time += dt
            
        # Extract position and velocities
        pos = ((f[0:nparticles] % L) + L) % L
        vel = f[nparticles:]
        
        # Send to output functions
        for func in out:
            func(pos, vel, ncells, L, time)
        
    return pos, vel

####################################################################
# 
# Output functions and classes
#

class Plot:
    """
    Displays three plots: phase space, charge density, and velocity distribution
    """
    def __init__(self, pos, vel, ncells, L):
        
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        # Plot initial positions
        if got_gridspec:
            self.fig = plt.figure()
            self.gs = gridspec.GridSpec(4, 4)
            ax = self.fig.add_subplot(self.gs[0:3,0:3])
            self.phase_plot = ax.plot(pos, vel, '.')[0]
            ax.set_title("Phase space")
            
            ax = self.fig.add_subplot(self.gs[3,0:3])
            self.density_plot = ax.plot(linspace(0, L, ncells), d)[0]
            
            ax = self.fig.add_subplot(self.gs[0:3,3])
            self.vel_plot = ax.plot(vhist, vbins)[0]
        else:
            self.fig = plt.figure()
            self.phase_plot = plt.plot(pos, vel, '.')[0]
            
            self.fig = plt.figure()
            self.density_plot = plt.plot(linspace(0, L, ncells), d)[0]
            
            self.fig = plt.figure()
            self.vel_plot = plt.plot(vhist, vbins)[0]
        plt.ion()
        plt.show()
        
    def __call__(self, pos, vel, ncells, L, t):
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        self.phase_plot.set_data(pos, vel) # Update the plot
        self.density_plot.set_data(linspace(0, L, ncells), d)
        self.vel_plot.set_data(vhist, vbins)
        plt.draw()
        plt.pause(0.05)
        

class Summary:
    def __init__(self):
        self.t = []
        self.firstharmonic = []
        
    def __call__(self, pos, vel, ncells, L, t):
        # Calculate the charge density
        d = calc_density(pos, ncells, L)
        
        # Amplitude of the first harmonic
        fh = 2.*abs(fft(d)[1]) / float(ncells)
        
        print(f"Time: {t} First: {fh}")
        
        self.t.append(t)
        self.firstharmonic.append(fh)

####################################################################
# 
# Functions to create the initial conditions
#

def landau(npart, L, alpha=0.2):
    """
    Creates the initial conditions for Landau damping
    
    """
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    pos0 = pos.copy()
    k = 2.*pi / L
    for i in range(10): # Adjust distribution using Newton iterations
        pos -= ( pos + alpha*sin(k*pos)/k - pos0 ) / ( 1. + alpha*cos(k*pos) )
        
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    return pos, vel

def twostream(npart, L, vbeam=2):
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    np2 = int(npart / 2)
    vel[:np2] += vbeam  # Half the particles moving one way
    vel[np2:] -= vbeam  # and half the other
    
    return pos,vel





####################################################################

if __name__ == "__main__":
    
    # --- user input for number of runs --- #

    N_runs = int(input("How many runs of the PIC simulation? "))
    
    # ensure directory exists
    result_dir = r"C:\Users\bpx519\OneDrive - University of York\Desktop\CompSci\Lab\Session23_results"
    os.makedirs(result_dir, exist_ok=True)
    
    # Arrays for stats across runs
    noise_list = []
    omega_list = []
    omega_err_list = []
    gamma_list = []
    gamma_err_list = []
    
    
    for run_idx in range(1, N_runs+1):
        print(f"\n==============================")
        print(f"        RUN {run_idx}/{N_runs}")
        print(f"==============================")
            
        start = time.perf_counter()
        
        # ----- Generate initial condition -----
        
        npart = 1000   
        if False:
            # 2-stream instability
            L = 100
            ncells = 20
            pos, vel = twostream(npart, L, 3.) # Might require more npart than Landau!
        else:
            # Landau damping
            L = 4.*pi
            ncells = 20
            pos, vel = landau(npart, L)
        
        # ----- Output Class -----
        
        #p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
        s = Summary()                 # Calculates, stores and prints summary info

        #diagnostics_to_run = [p, s]   # Remove p to get much faster code!
        diagnostics_to_run = [s]   # Remove p to get much faster code!
        
        # ----- Run the simulation -----
        pos, vel = run(pos, vel, L, ncells, 
                    out = diagnostics_to_run,        # These are called each output step
                    output_times=linspace(0.,20,50)) # The times to output
        


        # Summary stores an array of the first-harmonic amplitude
        # Make a semilog plot to see exponential damping

        plt.figure()
        plt.plot(s.t, s.firstharmonic)
        plt.xlabel("Time [Normalised]")
        plt.ylabel("First harmonic amplitude [Normalised]")
        plt.yscale('log')

        # Title includes the run ID
        plt.title(f"First Harmonic Damping (Run {run_idx})")

        # Ensure the output directory exists
        os.makedirs(result_dir, exist_ok=True)

        # Save with run ID appended
        outfile = os.path.join(result_dir, f"first_harmonic_run_{run_idx}.png")
        plt.savefig(outfile, dpi=300, bbox_inches='tight')

        plt.close()   # closes the figure so you don’t leak memory on repeated runs



        # Convert Summary Arrays
        t = np.array(s.t)
        A = np.array(s.firstharmonic) 
        
        # ----- ANALYSIS BLOCK -----

        # Find peaks
        peak_idx, _ = find_peaks(A) #The second output is metadata — ignored here (_).
        t_peaks = t[peak_idx]
        A_peaks = A[peak_idx]


        # Noise Detection
        noise_start_index = None
        for i in range(1, len(A_peaks)):
            if A_peaks[i] > A_peaks[i-1]:
                noise_start_index = i
                break

        if noise_start_index is None:
            print("Noise region never detected.")
            noise_start_index = len(A_peaks)

        # 'Split data into signal peaks + noise peaks'
        t_sig = t_peaks[:noise_start_index]          # Signal peaks : All peaks before noise -> actual physics.
        A_sig = A_peaks[:noise_start_index]

        t_noise = t_peaks[noise_start_index:]        # Noise peaks : Everything after ->  numerical noise.
        A_noise = A_peaks[noise_start_index:]

        # Noise lvl
        if len(A_noise) > 0:
            noise_level = np.mean(A_noise)   # 'compute noise metric'
        else:
            noise_level = np.nan

        # Frequency
        if len(t_sig) >= 2:
            periods = np.diff(t_sig)
            period_mean = np.mean(periods)
            period_err = np.std(periods)

            omega_est = 2*np.pi/period_mean
            omega_err = 2*np.pi*period_err/(period_mean**2)
        else:
            omega_est = omega_err = np.nan

        # Damping
        if len(A_sig) >= 2:
            logA = np.log(A_sig)
            coeff = np.polyfit(t_sig, logA, 1)
            gamma_est = -coeff[0]

            # error: std of residuals / sqrt(N)
            fit_residuals = logA - np.polyval(coeff, t_sig)
            gamma_err = np.std(fit_residuals) / np.sqrt(len(A_sig))
        else:
            gamma_est = gamma_err = np.nan

        # ---------- STORE RESULTS ----------
        noise_list.append(noise_level if noise_level is not None else np.nan)
        omega_list.append(omega_est if not np.isnan(omega_est) else np.nan)
        omega_err_list.append(omega_err if not np.isnan(omega_err) else np.nan)
        gamma_list.append(gamma_est if not np.isnan(gamma_est) else np.nan)
        gamma_err_list.append(gamma_err if not np.isnan(gamma_err) else np.nan)

        
        # ---------- SAVE RUN RESULTS ----------

        outpath = os.path.join(result_dir, f"run_{run_idx}.txt")

        with open(outpath, "w", encoding="utf-8-sig") as f:
            f.write(f"Run {run_idx}\n")
            f.write(f"Noise level: {noise_level}\n")
            f.write(f"Omega: {omega_est} ± {omega_err}\n")
            f.write(f"Gamma: {gamma_est} ± {gamma_err}\n")

        print(f"Saved: {outpath}")

        end = time.perf_counter()
        print(f"simulation runtime: {end - start : .3f} seconds")
        
    # ----- Final Stats -----
    
    def mean_std(x):
        return np.nanmean(x), np.nanstd(x, ddof=1)              #ignores NaNs

    noise_mean, noise_std = mean_std(noise_list)
    omega_mean, omega_std = mean_std(omega_list)
    gamma_mean, gamma_std = mean_std(gamma_list)

    print("\n\n==============================")
    print("      Final Stats")
    print("==============================")
    print(f"Noise   : mean={noise_mean:.4g}, std={noise_std:.4g}")
    print(f"Omega   : mean={omega_mean:.4g}, std={omega_std:.4g}")
    print(f"Gamma   : mean={gamma_mean:.4g}, std={gamma_std:.4g}")

    print("\n--- ERROR COMPARISON ---")
    print(f"Individual ω error : {np.mean(omega_err_list):.4g}")
    print(f"Run-to-run ω spread: {omega_std:.4g}")

    print(f"Individual γ error : {np.mean(gamma_err_list):.4g}")
    print(f"Run-to-run γ spread: {gamma_std:.4g}")

    print("\nLarger value = dominant source of uncertainty.")
    
    # saving final stats to txt file
    summary_path = r"C:\Users\bpx519\OneDrive - University of York\Desktop\CompSci\Lab\Session23_results\Simsummary.txt"
    with open(summary_path, "w", encoding = "utf-8-sig") as f:
        f.write("==============================\n")
        f.write("      Final Stats\n")
        f.write("==============================\n")
        f.write(f"Noise   : mean={noise_mean:.4g}, std={noise_std:.4g}\n")
        f.write(f"Omega   : mean={omega_mean:.4g}, std={omega_std:.4g}\n")
        f.write(f"Gamma   : mean={gamma_mean:.4g}, std={gamma_std:.4g}\n")
        f.write("\n--- ERROR COMPARISON ---\n")
        f.write(f"Individual ω error : {np.mean(omega_err_list):.4g}\n")
        f.write(f"Run-to-run ω spread: {omega_std:.4g}\n")
        f.write(f"Individual γ error : {np.mean(gamma_err_list):.4g}\n")
        f.write(f"Run-to-run γ spread: {gamma_std:.4g}\n")
        f.write("\nLarger value = dominant source of uncertainty.\n")

    print(f"\nSummary saved to: {summary_path}")

        
