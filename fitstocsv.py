from astropy.io import fits
import urllib.request
import csv

url = "https://lambda.gsfc.nasa.gov/data/foregrounds/ilc/9yr/wmap_ilc_9yr_peaks_v5.fits"
local_fits = "wmap_ilc_peaks.fits"

print("Downloading catalog...")
urllib.request.urlretrieve(url, local_fits)

print("Opening FITS...")
hdul = fits.open(local_fits)
data = hdul[1].data
columns = data.columns.names
print("Columns:", columns)

csv_out = "wmap_ilc_peaks.csv"
print("Writing CSV...")

with open(csv_out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    for row in data:
        writer.writerow([row[col] for col in columns])

print("Done:", csv_out)
