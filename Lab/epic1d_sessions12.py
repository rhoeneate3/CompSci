#!/usr/bin/env python3
#
# Electrostatic PIC code in a 1D cyclic domain
import numpy as np
from numpy import arange, concatenate, zeros, linspace, floor, array, pi
from numpy import sin, cos, sqrt, random, histogram, abs, sqrt, max
from scipy.signal import find_peaks
import matplotlib.pyplot as plt # Matplotlib plotting library
import time

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
    
    start = time.perf_counter()
    
    # Generate initial condition
    # 
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
    
    # Create some output classes
    #p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
    s = Summary()                 # Calculates, stores and prints summary info

    #diagnostics_to_run = [p, s]   # Remove p to get much faster code!
    diagnostics_to_run = [s]   # Remove p to get much faster code!
    
    # Run the simulation
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
    
        
    plt.ioff() # This so that the windows stay open
    plt.show()
    
    ####################################################################
    #   PEAK FINDING, NOISE ANALYSIS, FREQUENCY + DAMPING EXTRACTION
    ####################################################################

    from scipy.signal import find_peaks
    import numpy as np

    # NumPy arrays make indexing and maths easier.   '1. Load/reference harmonic data'
    t = np.array(s.t)
    A = np.array(s.firstharmonic) 

    # -----------------------------
    # 1. Find peaks
    # -----------------------------
    # '2. detect peaks'
    peak_idx, _ = find_peaks(A) #The second output is metadata — ignored here (_).
    t_peaks = t[peak_idx]
    A_peaks = A[peak_idx]

    # Plot to verify peaks
    plt.figure()
    plt.plot(t, A, label="First harmonic")                                # 'plot signal + peaks overlay'
    plt.plot(t_peaks, A_peaks, 'x', label="Detected peaks")               # 'plot signal + peaks overlay'
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Peaks on harmonic amplitude")
    plt.savefig(r'C:\Users\bpx519\OneDrive - University of York\Desktop\CompSci\Lab\NoiseAssessment')
    plt.show()

    # -----------------------------
    # 2. Identify where noise overtakes signal
    #    (first peak that increases vs previous one)
    # -----------------------------
    
    # 'Loop through peaks to find where noise dominates'
    noise_start_index = None
    for i in range(1, len(A_peaks)):
        if A_peaks[i] > A_peaks[i-1]:
            noise_start_index = i
            break

    if noise_start_index is None:
        print("Noise region never detected.")
        noise_start_index = len(A_peaks)

# 'Split data into signal peaks + noise peaks'
    # Signal peaks : All peaks before noise -> actual physics.
    t_sig = t_peaks[:noise_start_index]
    A_sig = A_peaks[:noise_start_index]

    # Noise peaks : Everything after ->  numerical noise.
    t_noise = t_peaks[noise_start_index:]
    A_noise = A_peaks[noise_start_index:]

    # -----------------------------
    # 3. Estimate noise amplitude level
    # -----------------------------
    if len(A_noise) > 0:
        noise_level = np.mean(A_noise)   # 'compute noise metric'
    else:
        noise_level = None

    print("\n--- NOISE ANALYSIS ---")
    print("Noise starts at peak index:", noise_start_index)
    print("Estimated noise amplitude:", noise_level)

    # -----------------------------
    # 4. Frequency estimate from peak spacing
    # -----------------------------
    if len(t_sig) >= 2:
        periods = np.diff(t_sig)
        period_mean = np.mean(periods)
        period_err = np.std(periods)

        omega_est = 2*np.pi/period_mean
        omega_err = 2*np.pi*period_err/(period_mean**2)
    else:
        omega_est = omega_err = None

    print("\n--- FREQUENCY MEASUREMENT ---")
    print(f"Estimated ω = {omega_est:.3f} ± {omega_err:.3f}")

    # -----------------------------
    # 5. Damping rate estimate via exponential fit
    #    A(t) = A0 * exp(-gamma t)
    #    => log(A) = log(A0) - gamma t
    # -----------------------------
    if len(A_sig) >= 2:
        logA = np.log(A_sig)
        coeff = np.polyfit(t_sig, logA, 1)
        gamma_est = -coeff[0]

        # error: std of residuals / sqrt(N)
        fit_residuals = logA - np.polyval(coeff, t_sig)
        gamma_err = np.std(fit_residuals) / np.sqrt(len(A_sig))
    else:
        gamma_est = gamma_err = None

    print("\n--- DAMPING RATE ---")
    print(f"Estimated γ = {gamma_est:.3f} ± {gamma_err:.3f}")

    # -----------------------------
    # 6. Comparison with expected
    # -----------------------------
    print("\n--- COMPARISON ---")
    print("Analytic:         ω = 1.416,  γ = 0.153")
    print("Typical PIC:      ω = 1.33 ± 0.16, γ = 0.168 ± 0.002")
    print()


    
    end = time.perf_counter()
    print(f"simulation runtime: {end - start : .3f} seconds")
    
