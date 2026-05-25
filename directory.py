from astropy.io import fits
import numpy as np
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic
import csv

# Load the FITS file
hdul = fits.open('wmap_ilc_9yr_v5.fits')
data = hdul[1].data
temp = data['TEMPERATURE']
nobs = data['N_OBS']

nside = 512
hp = HEALPix(nside=nside, order='ring', frame=Galactic())
pixel_index = np.arange(len(temp))

print("Finding neighbours and extracting extrema...")

# Get all 8 neighbours for every pixel
neighbours = hp.neighbours(pixel_index)

# Local max: temp > all 8 neighbours
# Local min: temp < all 8 neighbours
is_max = np.ones(len(temp), dtype=bool)
is_min = np.ones(len(temp), dtype=bool)

for i in range(8):
    nbr = neighbours[i]
    valid = nbr >= 0
    is_max &= valid & (temp >= temp[np.where(valid, nbr, 0)])
    is_min &= valid & (temp <= temp[np.where(valid, nbr, 0)])

# Only trust pixels observed more than once
quality = nobs > 1

max_pix = pixel_index[is_max & quality]
min_pix = pixel_index[is_min & quality]

print(f"Local maxima found: {len(max_pix)}")
print(f"Local minima found: {len(min_pix)}")

# Get coordinates
max_coords = hp.healpix_to_skycoord(max_pix)
min_coords = hp.healpix_to_skycoord(min_pix)

max_temps = temp[max_pix]
min_temps = temp[min_pix]

# Sort hottest to coldest
max_order = np.argsort(max_temps)[::-1]
min_order = np.argsort(min_temps)

# Save top 250 hot and top 250 cold to CSV
with open('cmb_extrema.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['type', 'l', 'b', 'T_mK'])
    for i in max_order[:250]:
        writer.writerow(['HOT', f"{max_coords.l.deg[i]:.3f}", f"{max_coords.b.deg[i]:.3f}", f"{max_temps[i]:.6f}"])
    for i in min_order[:250]:
        writer.writerow(['COLD', f"{min_coords.l.deg[i]:.3f}", f"{min_coords.b.deg[i]:.3f}", f"{min_temps[i]:.6f}"])

print("Done. Saved to cmb_extrema.csv")