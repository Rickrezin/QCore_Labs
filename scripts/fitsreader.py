from astropy.io import fits
import numpy as np

hdul = fits.open('wmap_ilc_9yr_v5.fits')
data = hdul[1].data
print(data.columns)
print(data[:5])