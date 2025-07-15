
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

# GENERAL PARAMETERS
realFiltersFilename = 'realFilters.csv'
w = np.arange(270, 781).reshape(-1, 1)
subpl_y = 1
subpl_x = 1

# PARAMETERS OF OPSINS & OIL DROPLETS TO SIMULATE
animal = 'Maratus'
chromophore = 'a1'
lambMax = [355, 504, 532, 565]
oMid = [np.nan, np.nan, np.nan, np.nan]
oCut = [np.nan, np.nan, np.nan, np.nan]
Clrs = [[0.8, 0, 1], [0, 0.6, 1], [0, 0.9, 0], [1, 0, 0]]

# ACHROMATIC CHANNEL
dblPhot = 0
dblOnly = 0
chromophoreDbl = 'a1'
lambMaxDbl = [568, 568]
oMidDbl = [468, np.nan]
oCutDbl = [452, np.nan]

OMT_avail = 0

# Load filters from CSV
filters = np.loadtxt(realFiltersFilename, delimiter=',')[:511, :]
numFilters = filters.shape[1]

# OPSIN & OIL DROPLET TEMPLATE CALCULATIONS
if dblOnly == 1:
    lambMax = []
    subpl_y = 1
    subpl_x = 1
    Clrs = [[0, 0, 0]]

vpig = np.zeros((len(w), len(lambMax) + dblPhot))

def calculate_opsin(S, chromophore, wavelength):
    if chromophore == 'a1':
        a = 0.8795 + 0.0459 * np.exp(-((S - 300) ** 2) / 11940)
        b = -40.5 + 0.195 * S
    elif chromophore == 'a2':
        a = 0.875 + 0.0268 * np.exp((S - 665) / 40.7)
        b = 317 - 1.149 * S + 0.00124 * (S ** 2)
    alpha = 1 / (np.exp(69.7 * (a - S / wavelength)) +
                 np.exp(28 * (0.922 - (S / wavelength))) +
                 np.exp(-14.9 * (1.104 - (S / wavelength))) + 0.674)
    beta = 0.26 * np.exp(-((wavelength - (189 + 0.315 * S)) / b) ** 2)
    return alpha + beta

for i, S in enumerate(lambMax):
    opsin = calculate_opsin(S, chromophore, w)
    if np.isnan(oMid[i]):
        vpig[:, i] = opsin[:, 0]
    else:
        oil = np.exp(-2.93 * np.exp(-2.89 * (0.5 / (oMid[i] - oCut[i])) * (w - oCut[i])))
        vpig[:, i] = (opsin * oil)[:, 0]
    if OMT_avail == 1:
        vpig[:, i] *= OMT[:, 0]

if dblPhot == 1:
    vpigDbl = np.zeros((len(w), len(lambMaxDbl)))
    for i, S in enumerate(lambMaxDbl):
        opsin = calculate_opsin(S, chromophoreDbl, w)
        if np.isnan(oMidDbl[i]):
            vpigDbl[:, i] = opsin[:, 0]
        else:
            oil = np.exp(-2.93 * np.exp(-2.89 * (0.5 / (oMidDbl[i] - oCutDbl[i])) * (w - oCutDbl[i])))
            vpigDbl[:, i] = (opsin * oil)[:, 0]
        if OMT_avail == 1:
            vpigDbl[:, i] *= OMT[:, 0]
    vpig[:, len(lambMax)] = np.sum(vpigDbl, axis=1)

vpig /= np.sum(vpig, axis=0)

coeffSet = []
synthOp = np.zeros_like(vpig)

plt.figure()
for i in range(vpig.shape[1]):
    res = lsq_linear(filters, vpig[:, i], bounds=(0, np.inf), method='trf')
    coeff = res.x
    coeffSet.append(coeff)
    synthOp[:, i] = np.dot(filters, coeff)
    spl_i = i + 1 if (subpl_y > 1 or subpl_x > 1) else 1
    plt.subplot(subpl_y, subpl_x, spl_i)
    plt.plot(w, vpig[:, i], color=Clrs[i], linewidth=3)
    plt.plot(w, synthOp[:, i], linestyle=':', color=Clrs[i], linewidth=3)
    plt.yticks([])
    if i >= subpl_x * (subpl_y - 1):
        plt.xticks(np.arange(300, 701, 100))
    else:
        plt.xticks([])
    plt.xlim([300, 720])
    if subpl_y > 1 or subpl_x > 1:
        plt.title(str(lambMax[i]))
    plt.gca().tick_params(labelsize=14)

plt.xlabel('wavelength (nm)', fontsize=12)
plt.gcf().set_size_inches(9, 3.5)
plt.xlim([290, 710])
plt.tight_layout()
plt.savefig(f'{animal}ComputatFilters.png', dpi=600)

# Save coefficients to CSV
np.savetxt(f'{animal}ComputatFilterCoeffs.csv', np.array(coeffSet).T, delimiter=',')