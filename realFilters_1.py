# User-adjustable variables
inputFileDir = '/Users/ctedore/Documents/GitHub/16-filter-camera/' # directory where all code and input files are located 
outputFileDir = '/Users/ctedore/Documents/Coding Projects/16-filter-camera/' # directory you want files to be written to
cameraSpecFile = 'cameraComponentsSpectra.csv' # name of .csv file with camera lens transmittance spectrum and camera sensor quantum efficiency spectrum
filterFilesBaseName = 'Transmission' # first part of bandpass filter filenames name that is common to all of them

# Packages to import
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

# Camera component spectra
camera = pd.read_csv(inputFileDir + cameraSpecFile, sep=';')

# Set wavelength range
wavelengths = np.arange(270, 992)
num_points = len(wavelengths)

# List transmission files 
files = sorted([f for f in os.listdir(inputFileDir) if f.startswith(filterFilesBaseName) and f.endswith('.txt')])
numfiles = len(files)

# Assign colors for plotting
colors = np.vstack([np.tile([1,0,0.95],(2,1)), np.tile([0.62,0,1],(2,1)), np.tile([0,1,1],(2,1)), np.tile([0.05,0.65,0.59],(2,1)), np.tile([0.46,0.94,0.36],(2,1)), np.tile([0.81,0.81,0.04],(2,1)), np.tile([0.93,0.69,0.13],(2,1)), np.tile([1,0,0],(2,1))])

# Plot setup
plt.figure(figsize=(9, 3.5))

filters = np.zeros((num_points, numfiles))
for i, filename in enumerate(files):
    with open(filename) as f:
        lines = f.readlines()[100:]  # Skip header
        data = np.genfromtxt(lines, max_rows=960)

    # Interpolate and apply camera component spectra
    interpFunc = interp1d(data[:, 0], data[:, 1], bounds_error=False, fill_value=0)
    y = interpFunc(wavelengths) / 100.0
    y *= camera.lens * camera.cameraSensor
    filters[:, i] = y

    style = '-' if i % 2 == 0 else ':'
    plt.plot(wavelengths, y, style, linewidth=3, color=colors[i])

# Plot formatting
plt.xlim([290, 710])
plt.ylim([0, 0.65])
plt.xlabel('wavelength (nm)', fontsize=12)
plt.ylabel('relative sensitivity', fontsize=12)
# plt.yticks([])
plt.xticks(fontsize=12)
plt.tight_layout()

# Save outputs
plt.savefig(outputFileDir + 'realFilters.png', dpi=600)
np.savetxt(outputFileDir + 'realFilters.csv', filters, delimiter=',')
print("Processing complete. Outputs saved as 'realFilters.png' and 'realFilters.csv'.")