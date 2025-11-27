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

def slice_into_chunks(lst, chunk_size):                     #helper function
    """Turn [a,b,c,a,b,c,...] into [[a,b,c],[a,b,c],...]"""
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]




####################################################################

if __name__ == "__main__":

    # -------------------------------------------------------------
    # USER INPUT
    # -------------------------------------------------------------
    N_runs = int(input("How many runs of the PIC simulation? "))
    mode = input("Sweep over 'cells', 'particles', or 'box_length'? here, box_length will be (input)*np.pi ").strip().lower()

    if mode not in ("cells", "particles", "box_length"):
        raise ValueError("You must choose either 'cells', 'particles', or 'box_length'.")

    values = [int(v) for v in input(
        f"Enter values of {mode} separated by commas: "
    ).split(",")]

    # directory setup
    result_dir = r"C:\Users\bpx519\OneDrive - University of York\Desktop\CompSci\Lab\Session23_results"
    os.makedirs(result_dir, exist_ok=True)

    # -------------------------------------------------------------
    # STORAGE (one big list per metric)
    # -------------------------------------------------------------
    noise_list = []
    omega_list = []
    omega_err_list = []
    gamma_list = []
    gamma_err_list = []
    sim_time_list = []

    # -------------------------------------------------------------
    # MAIN SWEEP LOOP
    # -------------------------------------------------------------
    for v in values:
        print(f"\n##### Sweeping {mode} = {v} #####")

        for run_idx in range(1, N_runs + 1):

            # ----------------------------------------------
            # set swept variable only here
            # ----------------------------------------------
            if mode == "cells":
                ncells = v
                npart = 1000
                L = 4 * np.pi
            elif mode == "particles":
                npart = v
                ncells = 20
                L = 4 * np.pi
            else:
                L = v * np.pi
                ncells = 40
                npart = 2000
            # ----------------------------------------------
            # Initial condition
            # ----------------------------------------------
            pos, vel = landau(npart, L)

            s = Summary()
            diagnostics_to_run = [s]

            # ----------------------------------------------
            # TIME THE SIMULATION
            # ----------------------------------------------
            start = time.perf_counter()
            pos, vel = run(
                pos, vel, L, ncells,
                out=diagnostics_to_run,
                output_times=np.linspace(0., 20., 50)
            )
            sim_time = time.perf_counter() - start

            # ----------------------------------------------
            # ANALYSIS BLOCK (unchanged from your code)
            # ----------------------------------------------
            t = np.array(s.t)
            A = np.array(s.firstharmonic)

            peak_idx, _ = find_peaks(A)
            t_peaks = t[peak_idx]
            A_peaks = A[peak_idx]

            # detect noise region
            noise_start = None
            for i in range(1, len(A_peaks)):
                if A_peaks[i] > A_peaks[i-1]:
                    noise_start = i
                    break
            if noise_start is None:
                noise_start = len(A_peaks)

            t_sig = t_peaks[:noise_start]
            A_sig = A_peaks[:noise_start]
            A_noise = A_peaks[noise_start:]

            noise_level = np.mean(A_noise) if len(A_noise) else np.nan

            # frequency
            if len(t_sig) >= 2:
                dT = np.diff(t_sig)
                meanT = np.mean(dT)
                stdT  = np.std(dT)
                omega = 2 * np.pi / meanT
                omega_err = 2 * np.pi * stdT / (meanT**2)
            else:
                omega = omega_err = np.nan

            # damping
            if len(A_sig) >= 2:
                logA = np.log(A_sig)
                coeff = np.polyfit(t_sig, logA, 1)
                gamma = -coeff[0]
                residuals = logA - np.polyval(coeff, t_sig)
                gamma_err = np.std(residuals) / np.sqrt(len(A_sig))
            else:
                gamma = gamma_err = np.nan

            # ----------------------------------------------
            # STORE RESULTS FOR LATER PLOTTING
            # ----------------------------------------------
            noise_list.append(noise_level)
            omega_list.append(omega)
            omega_err_list.append(omega_err)
            gamma_list.append(gamma)
            gamma_err_list.append(gamma_err)
            sim_time_list.append(sim_time)

        # end run loop

    # end sweep loop

    # -------------------------------------------------------------
    # GROUP DATA INTO CHUNKS (one chunk per swept value)
    # -------------------------------------------------------------
    def chunk(lst):
        return [lst[i:i+N_runs] for i in range(0, len(lst), N_runs)]

    noise_chunks = chunk(noise_list)
    omega_chunks = chunk(omega_list)
    omega_err_chunks = chunk(omega_err_list)
    gamma_chunks = chunk(gamma_list)
    gamma_err_chunks = chunk(gamma_err_list)
    time_chunks  = chunk(sim_time_list)

    # -------------------------------------------------------------
    # COMPUTE MEANS + STD (Error bars)
    # -------------------------------------------------------------
    noise_means = [np.nanmean(c) for c in noise_chunks]
    noise_stds  = [np.nanstd(c, ddof=1) for c in noise_chunks]

    omega_means = [np.nanmean(c) for c in omega_chunks]
    omega_stds  = [np.nanstd(c, ddof=1) for c in omega_chunks]

    gamma_means = [np.nanmean(c) for c in gamma_chunks]
    gamma_stds  = [np.nanstd(c, ddof=1) for c in gamma_chunks]

    time_means = [np.nanmean(c) for c in time_chunks]
    time_stds  = [np.nanstd(c, ddof=1) for c in time_chunks]

    # -------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------
    xlabel = "Number of Cells" if mode == "cells" else "Number of Particles"

    def plot_with_error(y, yerr, ylabel, fname):
        plt.figure()
        plt.errorbar(values, y, yerr=yerr, fmt="o-", capsize=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs {xlabel}")
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, fname), dpi=300)
        plt.close()

    plot_with_error(time_means,  time_stds,  "Simulation Time [s]", f"time_vs_{mode}.png")
    plot_with_error(noise_means, noise_stds, "Noise Level",         f"noise_vs_{mode}.png")
    plot_with_error(omega_means, omega_stds, "Frequency ω",         f"omega_vs_{mode}.png")
    plot_with_error(gamma_means, gamma_stds, "Damping γ",           f"gamma_vs_{mode}.png")

    print("\nAll plots saved.")
        # -------------------------------------------------------------
    # SUMMARY TEXT FILE (FULL RUN-BY-RUN DATA + ERRORS)
    # -------------------------------------------------------------
    summary_path = os.path.join(result_dir, f"Simsummary_sweep_{mode}.txt")

    with open(summary_path, "w") as f:
        f.write(f"# PIC Simulation Summary (sweep over {mode})\n")
        f.write(f"# Values swept: {values}\n")
        f.write(f"# Runs per configuration: {N_runs}\n\n")

        for idx, v in enumerate(values):
            # pull chunks for this config
            noise_c = noise_chunks[idx]
            omega_c = omega_chunks[idx]
            omega_err_c = omega_err_chunks[idx]
            gamma_c = gamma_chunks[idx]
            gamma_err_c = gamma_err_chunks[idx]
            time_c  = time_chunks[idx]

            # header for this block
            f.write(f"# config: {mode}={v}\n")
            if mode == "cells":
                f.write(f"# Ncells={v}  Nparticles=1000\n")
            else:
                f.write(f"# Ncells=20   Nparticles={v}\n")

            f.write("run_id  runtime   noise   freq   freq_err   damping   damping_err\n")

            # individual runs
            for run_id in range(N_runs):
                f.write(
                    f"{run_id+1:<7d}"
                    f"{time_c[run_id]:<10.5g} "
                    f"{noise_c[run_id]:<7.5g} "
                    f"{omega_c[run_id]:<7.5g} "
                    f"{omega_err_c[run_id]:<10.5g} "
                    f"{gamma_c[run_id]:<10.5g} "
                    f"{gamma_err_c[run_id]:<10.5g}\n"
                )

            # summary statistics
            f.write("\n# mean values:\n")
            f.write(
                f"# runtime_mean={time_means[idx]:.6g}, runtime_std={time_stds[idx]:.6g}\n"
            )
            f.write(
                f"# noise_mean={noise_means[idx]:.6g}, noise_std={noise_stds[idx]:.6g}\n"
            )
            f.write(
                f"# freq_mean={omega_means[idx]:.6g}, freq_std={omega_stds[idx]:.6g}\n"
            )
            f.write(
                f"# damping_mean={gamma_means[idx]:.6g}, damping_std={gamma_stds[idx]:.6g}\n"
            )
            f.write("\n" + "-"*70 + "\n\n")

    print(f"\nSummary saved to: {summary_path}")
    
    
    # ---------------------------
# FIT: noise ~ A * x^p (expect p = -0.5 for 1/sqrt(Np_per_cell))
# ---------------------------
import numpy as _np
from scipy.optimize import curve_fit

# determine effective x variable = particles per cell for each swept 'values'
# Use the same defaults you used in the sweep:
DEFAULT_NPART_FOR_CELL_SWEEP = 1000
DEFAULT_NCELLS_FOR_PARTICLE_SWEEP = 20

if mode == "cells":
    Ncells_arr = _np.array(values, dtype=float)
    Npart_arr = _np.full_like(Ncells_arr, DEFAULT_NPART_FOR_CELL_SWEEP, dtype=float)
elif mode == "particles":
    Npart_arr = _np.array(values, dtype=float)
    Ncells_arr = _np.full_like(Npart_arr, DEFAULT_NCELLS_FOR_PARTICLE_SWEEP, dtype=float)
else:
    # for box_length sweeps we'll just fit noise vs particles-per-cell using defaults
    Ncells_arr = _np.full(len(values), 40.0)    # whatever default you used when sweeping box_length
    Npart_arr  = _np.full(len(values), 2000.0)

# effective independent variable: particles per cell
x = Npart_arr / Ncells_arr    # N_particles per cell
y = _np.array(noise_means, dtype=float)
yerr = _np.array(noise_stds, dtype=float)

# mask invalid / nan entries
mask = _np.isfinite(x) & _np.isfinite(y)
x = x[mask]; y = y[mask]; yerr = yerr[mask]

# avoid zero yerr (replace zeros with a small floor)
yerr = _np.where(yerr <= 0, y.max()*1e-1 + 1e-12, yerr)

def power_model(x, A, p):
    return A * x**p

fit_ok = False
if len(x) >= 2:
    try:
        # initial guess: A ~ y * sqrt(x) (if p ~ -0.5)
        p0 = [-0.5]   # starting guess for exponent
        A0 = _np.nanmean(y * _np.sqrt(x))
        popt, pcov = curve_fit(power_model, x, y, p0=[A0, -0.5], sigma=yerr, absolute_sigma=True, maxfev=10000)
        A_best, p_best = popt
        perr = _np.sqrt(_np.diag(pcov))
        A_err, p_err = perr
        fit_ok = True
    except Exception as e:
        print("Fit failed:", e)
        A_best = _np.nan; p_best = _np.nan; A_err = _np.nan; p_err = _np.nan
else:
    A_best = _np.nan; p_best = _np.nan; A_err = _np.nan; p_err = _np.nan

print(f"\nFit (noise ≈ A * (Npart/Ncell)^p): A = {A_best:.4g} ± {A_err:.4g}, p = {p_best:.4g} ± {p_err:.4g}")

# Save a diagnostic plot
xx = _np.logspace(_np.log10(x.min()*0.9), _np.log10(x.max()*1.1), 200)
plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label='data')
if fit_ok:
    plt.plot(xx, power_model(xx, A_best, p_best), '-', label=f'fit: A={A_best:.3g}, p={p_best:.3g}')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Particles per cell (Npart / Ncells)')
plt.ylabel('Noise level')
plt.title('Noise vs particles-per-cell (log-log)')
plt.legend(); plt.grid(True, which='both')
plt.savefig(os.path.join(result_dir, f"noise_fit_{mode}_particlespercell.png"), dpi=300)
plt.close()
