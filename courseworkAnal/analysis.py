# A simple example of a serial processing script which we want to accelerate.

#from joblib import Parallel, delayed #You'll want to use these to accelerate the job

from scipy.signal import correlate
from numpy import loadtxt
import matplotlib.pyplot as plt
from time import time

def process_data(data_slice, frame = 0, save_figure = False):
    if save_figure:
        plt.clf()
        plt.pcolormesh(data_slice) ; plt.colorbar()
        plt.title(f"Time index : {frame:03d}")
        plt.savefig(f"input_frame_{frame:03d}.png", dpi = 150)
        
    # Find a form of auto-correlation
    result = correlate(data_slice, data_slice,
                       mode = 'full', method = 'direct')

    if save_figure:
        plt.clf()
        plt.pcolormesh(result) ; plt.colorbar()
        plt.title(f"Time index : {frame:03d}")
        plt.savefig(f"frame_{frame:03d}.png", dpi = 150)

    return result

def serial_processing(data, operation, **kwargs):
    start = time()
    for i in range(0, data.shape[0]):
        operation(data[i,:,:].squeeze(), frame = i, **kwargs)
    end = time()
    return end - start


if __name__ == "__main__":
    # Let us read in the full data
    data = loadtxt("vort_reduced.txt").reshape([65, 65, 64])
    elapsed_time = serial_processing(data, process_data, save_figure = True)
    print(f"Processing took {elapsed_time}s")
