#!/usr/bin/env python

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.io import fits
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as mtick

if(len(sys.argv) != 4):
	sys.stderr.write("Usage: mask_stats.py <ALMA data cube> <ALMA mask cube> <SoFiA mask cube>.\n");
	sys.exit(1);

sys.stderr.write("\nComparison of ALMA and SoFiA mask and ALMA mask statistics.\n\n");

# Read FITS files
try:
	hdu  = fits.open(sys.argv[1]);
	head = hdu[0].header;
	data = hdu[0].data;
except:
	sys.stderr.write("Error: Failed to read data cube {:s}.\n".format(sys.argv[1]));
	sys.exit(1);

try:
	hdu_alma   = fits.open(sys.argv[2]);
	head_alma  = hdu_alma[0].header;
	mask_alma  = hdu_alma[0].data;
except:
	sys.stderr.write("Error: Failed to read ALMA mask cube {:s}.\n".format(sys.argv[2]));
	sys.exit(1);

try:
	hdu_sofia  = fits.open(sys.argv[3]);
	head_sofia = hdu_sofia[0].header;
	mask_sofia = hdu_sofia[0].data;
except:
	sys.stderr.write("Error: Failed to read SoFiA mask cube {:s}.\n".format(sys.argv[3]));
	sys.exit(1);

# Sanity checks
dim_data       = len(data.shape);
dim_mask_alma  = len(mask_alma.shape);
dim_mask_sofia = len(mask_sofia.shape);

assert(dim_data       > 1 and dim_data       < 5);
assert(dim_mask_alma  > 1 and dim_mask_alma  < 5);
assert(dim_mask_sofia > 1 and dim_mask_sofia < 5);

assert(dim_data       < 4 or data.shape[0]       == 1);
assert(dim_mask_alma  < 4 or mask_alma.shape[0]  == 1);
assert(dim_mask_sofia < 4 or mask_sofia.shape[0] == 1);

if(dim_data       > 3):
	sys.stderr.write("Warning: Dropping 4th axis of data cube.\n");
	data       = np.reshape(data,       data.shape[1:]);
if(dim_mask_alma  > 3):
	sys.stderr.write("Warning: Dropping 4th axis of ALMA mask.\n");
	mask_alma  = np.reshape(mask_alma,  mask_alma.shape[1:]);
if(dim_mask_sofia > 3):
	sys.stderr.write("Warning: Dropping 4th axis of SoFiA mask.\n");
	mask_sofia = np.reshape(mask_sofia, mask_sofia.shape[1:]);

# Determine number of SoFiA sources
n_src = int(np.nanmax(mask_sofia));
if(n_src < 1):
	sys.stderr.write("Error: No sources found in SoFiA mask.\n");
	sys.exit(1);

sys.stdout.write("\nFound {0:d} sources in SoFiA mask.\n\n".format(n_src));

# Loop over all sources
for src in range(1, n_src + 1):
	data_masked = data[mask_sofia == src];
	npix_sofia  = np.nansum(mask_sofia == src);
	npix_alma   = np.nansum(mask_alma[mask_sofia == src] > 0);
	flux_sofia  = np.nansum(data_masked);
	flux_alma   = np.nansum(data_masked[mask_alma[mask_sofia == src] > 0]);
	
	sys.stdout.write("Source {:d}:\n  N_SoFiA = {:d}\n  N_ALMA  = {:d}\n  F_SoFiA = {:.2f}\n  F_ALMA  = {:.2f}\n".format(src, npix_sofia, npix_alma, flux_sofia, flux_alma));
	sys.stdout.write("  ALMA mask pixel fraction: {:.2f}%\n  ALMA mask flux fraction:  {:.2f}%\n".format(100.0 * npix_alma / npix_sofia, 100.0 * flux_alma / flux_sofia));

# determine dimensions of the ALMA cube
npix1 = mask_alma.shape[0];
npix2 = mask_alma.shape[1];
npix3 = mask_alma.shape[2];
sys.stdout.write("\nALMA Cube Dimension: {:d} by {:d} by {:d} pixels \n".format(npix3, npix2, npix1));
        
# determine flagged fraction per channel
nr_pix_mask = np.count_nonzero(~np.isnan(mask_alma));
nr_pix_data = np.count_nonzero(~np.isnan(data));
nr_masked_pix = np.count_nonzero(mask_alma == 1);
fraction_data_pix = 100 * (nr_pix_data / nr_pix_mask);
fraction_masked_pix = 100 * (nr_masked_pix / nr_pix_data);
# print (nr_pix_data, nr_pix_mask, nr_masked_pix, fraction_masked_pix);
sys.stdout.write("\nALMA Mask Statistics: \n  valid data pixels:  {:d}\n  total pixels:  {:d}\n  fraction of valid data pixels =  {:.2f} %\n  masked pixels:  {:d}\n  fraction of flagged pixels =  {:.2f} %\n\n For detailed statistics per channel see the files:\n mask_statistics.csv and mask_statistics.pdf\n".format(nr_pix_data, nr_pix_mask, fraction_data_pix, nr_masked_pix, fraction_masked_pix));

# initialise array to store fractions
fraction_masked_pix_per_channel = np.zeros(npix1);
#print (fraction_masked_pix_per_channel);

# calculate the fraction of flagged pixels and write it to a csv table
nr_pix_mask_per_channel = np.count_nonzero((mask_alma == 1), axis=(1, 2));
nr_pix_data_per_channel = np.count_nonzero(~np.isnan(data), axis=(1, 2));
replacement_value = np.full(nr_pix_mask_per_channel.shape, np.nan, dtype=float);

fraction_masked_pix_per_channel = np.divide(nr_pix_mask_per_channel, nr_pix_data_per_channel, out=replacement_value, where = nr_pix_data_per_channel != 0)

#print (nr_pix_mask_per_channel, "\n", nr_pix_data_per_channel, "\n", fraction_masked_pix_per_channel);

table = np.column_stack((np.arange(len(fraction_masked_pix_per_channel)), fraction_masked_pix_per_channel));
np.savetxt(
    "mask_statistics.csv",
    table,
    delimiter=",",
    header="#Channel,#Value",
    comments="",
    fmt=["%d", "%.6f"] 
    );

# make a plot
plt.figure(figsize=(10,4))
plt.autoscale(enable=True, axis='both', tight=None)
plt.plot(fraction_masked_pix_per_channel, 'r-', label='Spectrum')
plt.plot([1, npix1], [0, 0],'k--')
plt.xlabel('Channel', fontsize = 12)
plt.ylabel('Fraction of flagged pixels', fontsize = 12)
plt.savefig('mask_statistics.pdf',format = 'pdf', bbox_inches = 'tight', transparent=True)

# close fits files
hdu.close();
hdu_alma.close();
hdu.close();
