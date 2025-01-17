#!/usr/bin/python3

# Analysis of spectral data and bathymetry
# This script creates a grid of square cells over the provided Landsat image set.
# For each cell, we merge bathymetry data with spectral data.
# For areas where bathymetry<0 (under sea level), we compute NDVI.
# Finally, we output a CSV file containing a list of lat, lon, NDVI - one line per cell in the grid.
# Lat, lon corresponds to the center of the cell.
# Only cells in the specified NDVI quantile are returned.

# Landsat Images can be downloaded from https://earthexplorer.usgs.gov

# note: code optimized for Python 3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os

# on the mac, install with 'conda install gdal'
from osgeo import osr, gdal

# get command line parameters
parser = argparse.ArgumentParser(description='Calculate NDVI on Landsat images and return list of areas with kelp.')
parser.add_argument("landsat", help="path to dir containing a full set of Landsat image bands")
parser.add_argument("bathymetry", help="path to file containing bathymetry data")
parser.add_argument("output", help="path for resulting CSV dataset")
parser.add_argument("--bathymetry2", help="additional path to file containing more bathymetry data")
parser.add_argument("-g", "--gridsize", help="number of cells per side in grid", type=int, default=300)
parser.add_argument("-q", "--quantile", help="quantile of extracted NDVI", type=float, default=0.95)
parser.add_argument("-t", "--threshold", help="NDVI threshold", type=float)
parser.add_argument("-i", "--image", help="generate image", action="store_true")
parser.add_argument("-m", "--mindepth", help="min depth of water for detection", type=float, default=-5)
parser.add_argument("-cx", "--correction_x", help="x correction in pixels to achieve precise alignment between bathymetry and spectral images", type=int, default=0)
parser.add_argument("-cy", "--correction_y", help="y correction in pixels to achieve precise alignment between bathymetry and spectral images", type=int, default=0)
parser.add_argument("-x", "--max", help="use max ndvi for esch cell (as opposed to mean)", action="store_true")
args = parser.parse_args()


# load images for bands 4 and 5 + the color image
# note: matplotlib can only handle png natively
# 'pip install pillow' to add support for more image types
print("-------------")
print("Kelp detector")
print("-------------")

landsat_dir_name = args.landsat
all_bands = os.listdir(landsat_dir_name)

b4_file = [fn for fn in all_bands if fn.endswith("_B4.TIF")][0]
b5_file = [fn for fn in all_bands if fn.endswith("_B5.TIF")][0]
color_file = [fn for fn in all_bands if fn.endswith("_T1.jpg")][0]

has_mask = False
mask_files = [fn for fn in all_bands if fn.endswith("_MASK.TIF")]
if len(mask_files)>0:
    has_mask = True
    mask_file = mask_files[0]


print("Reading Band 4 file: " + b4_file)
print("Reading Band 5 file: " + b5_file)
print("Reading Color file: " + color_file)

if has_mask:
    print("Reading Mask file: " + mask_file)

print("Quantile: " + str(args.quantile))
print("Threshold: " + str(args.threshold))
print("Min depth: " + str(args.mindepth))
print("Correction X: " + str(args.correction_x))
print("Correction Y: " + str(args.correction_y))

stat = "mean"
if args.max:
    stat="max"

print("NDVI aggregation: " + stat)

###########################

b4 = plt.imread(os.path.join(landsat_dir_name, b4_file)) # red
b5 = plt.imread(os.path.join(landsat_dir_name, b5_file)) # infrared

if has_mask:
    mask = plt.imread(os.path.join(landsat_dir_name, mask_file)) # mask
    new_mask = np.ones(mask.shape)
    np.place(new_mask, mask == 0, np.nan) # mask with 1s in valid pixels, nan where we must ignore

color_img = plt.imread(os.path.join(landsat_dir_name, color_file))

# get the existing coordinate system
ds = gdal.Open(os.path.join(landsat_dir_name, b4_file))

old_cs = osr.SpatialReference()
old_cs.ImportFromWkt(ds.GetProjectionRef())

# create the new coordinate system
wgs84_wkt = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
new_cs = osr.SpatialReference()
new_cs.ImportFromWkt(wgs84_wkt)

# create a transform object to convert between coordinate systems
transform = osr.CoordinateTransformation(old_cs, new_cs)
gt = ds.GetGeoTransform()

def getLonLat(x, y):
    coord_x = gt[0] + x * gt[1]
    coord_y = gt[3] + y * gt[5]

    # get the coordinates in lat long
    latlong = transform.TransformPoint(coord_x, coord_y, 0.0)
    return latlong[0], latlong[1]

def getLat(x, y):
    return getLonLat(x, y)[1]

def getLon(x, y):
    return getLonLat(x, y)[0]


###############################

image_height = color_img.shape[0]
image_width = color_img.shape[1]

grid_size_h = args.gridsize
grid_size_v = args.gridsize

cell_size_pixel_h = int(image_width / grid_size_h)
cell_size_pixel_v = int(image_height / grid_size_v)

# how many lon/lat degrees

# adjust boundaries so there's an integer number of cells in the grid
image_width = cell_size_pixel_h * grid_size_h
image_height = cell_size_pixel_v * grid_size_v

# approximate tile size in degrees lat/lon
x0, y0 = getLonLat(0,0)
x1, y1 = getLonLat(cell_size_pixel_h, cell_size_pixel_v)
cell_size_deg_v = y0-y1
cell_size_deg_h = x1-x0

print("-------------")
print("Image size: " + str(image_width) + " by " + str(image_height))
print("Grid size: " + str(grid_size_h) + " by " + str(grid_size_v))
print("Pixel cell size: " + str(cell_size_pixel_h) + " by " + str(cell_size_pixel_v))
print("Approx. lon cell size: " + str(cell_size_deg_h))
print("Approx. lat cell size: " + str(cell_size_deg_v))

print("-------------")

b4 = b4[0:image_height, 0:image_width]
b5 = b5[0:image_height, 0:image_width]
color = color_img[0:image_height, 0:image_width].copy()

if has_mask:
    new_mask = new_mask[0:image_height, 0:image_width]
##############################

# ## Create grid
print("Creating grid...")
grid = pd.DataFrame(index=np.arange(0, grid_size_h * grid_size_v),
                    columns=["i", "j", "top", "left", "bottom", "right", "pxtop", "pxleft", "pxbottom", "pxright"])

grid["i"] = np.arange(0, grid_size_v).repeat(grid_size_h)
grid["j"] = np.tile(np.arange(0, grid_size_h), grid_size_v)

grid["pxleft"] = grid["j"] * cell_size_pixel_h
grid["pxright"] = grid["pxleft"] + cell_size_pixel_h - 1
grid["pxtop"] = grid["i"] * cell_size_pixel_v
grid["pxbottom"] = grid["pxtop"] + cell_size_pixel_v - 1

left = np.arange(0, image_width, cell_size_pixel_h)
top = np.arange(0, image_height, cell_size_pixel_v)
right = left + cell_size_pixel_h
bottom = top + cell_size_pixel_v

left_lon = [getLon(x, 0) for x in left]
top_lat = [getLat(0, y) for y in top]

right_lon = [getLon(x, 0) for x in right]
bottom_lat = [getLat(0, y) for y in bottom]

grid["top"] = np.array(top_lat).repeat(grid_size_h)
grid["left"] = np.tile(left_lon, grid_size_v)

grid["bottom"] = np.array(bottom_lat).repeat(grid_size_h)
grid["right"] = np.tile(right_lon, grid_size_v)

## Load bathymetry data
print("Loading bathymetry data...")

bathymetry_file = args.bathymetry
depth = xr.open_dataset(bathymetry_file).to_dataframe()
depth = depth.reset_index().dropna()
depth.columns = ["lon", "lat", "depth"]

if args.bathymetry2:
    print("Loading additional bathymetry data...")
    depth2 = xr.open_dataset(args.bathymetry2).to_dataframe()
    depth2 = depth2.reset_index().dropna()
    depth2.columns = ["lon", "lat", "depth"]
    depth = depth.append(depth2)

depth["j"] = np.searchsorted(left_lon, depth["lon"]) + args.correction_x
depth["i"] = np.searchsorted(-1 * np.array(top_lat), -1 * depth["lat"]) + args.correction_y
filtered_depth = pd.DataFrame(
    depth[(depth["i"] > 0) & (depth["j"] > 0) & (depth["i"] <= grid_size_v) & (depth["j"] <= grid_size_h)])
filtered_depth["idx"] = filtered_depth["i"] * grid_size_h + filtered_depth["j"]

# aggregate depth data based on the grid defined above
d = pd.DataFrame(filtered_depth.groupby(['idx'])['depth'].agg(['max', 'mean']))  # max and mean depth in cell
d.columns = ["max_depth", "mean_depth"]
grid = grid.join(d)


# compute NDVI based on bands B4 (red) and B5 (infrared)
print("Computing NDVI...")

ir_float = b5.astype(float)
red_float = b4.astype(float)
den = ir_float + red_float
den[den == 0] = np.nan
ndvi = np.divide(ir_float - red_float, den)
if has_mask:
    ndvi = np.multiply(ndvi, new_mask)

# create a new dataframe to hold the ndvi value for each cell in the grid
# to get the ndvi for each cell, we compute the mean or max ndvi value for all image pixels
# that fall in that cell
ndvi_grid = pd.DataFrame(index=np.arange(0, grid_size_h * grid_size_v), columns=["mean_ndvi"])

for i in range(0, grid_size_v):
    for j in range(0, grid_size_h):
        # find the boundaries (in pixel coordinates) of this grid cell
        pxleft = j * cell_size_pixel_h
        pxright = pxleft + cell_size_pixel_h - 1
        pxtop = i * cell_size_pixel_v
        pxbottom = pxtop + cell_size_pixel_v - 1
        # extract all image pixels within the boundaries
        myslice = ndvi[pxtop:pxbottom, pxleft:pxright].reshape(-1)
        # average the NDVI fot the extracted pixels
        mymean = 0
        if args.max:
            mymean = np.max(myslice)
        else:
            mymean = np.mean(myslice)
        # save value in the NDVI dataframe at the corresponding location
        ndvi_grid.loc[i * grid_size_h + j] = [mymean]

print("Finding kelp...")

grid = grid.join(ndvi_grid)
grid["mean_ndvi"] = grid["mean_ndvi"].astype(float)
ndvi_values = pd.DataFrame(grid[grid["max_depth"] < args.mindepth]).dropna()
filter_ndvi = args.threshold
if filter_ndvi==None:
    filter_ndvi = ndvi_values["mean_ndvi"].quantile(args.quantile)

print("NDVI threshold: " + str(filter_ndvi))


if args.image:
    # overlay grid on image
    print("Saving image...")

    # highlight cells with ndvi>filter_ndvi and depth<0
    def highlight_cell(image, color, row, force=False):
        cell = image[row.pxtop:row.pxbottom, row.pxleft:row.pxright]
        if force:
            cell = color
        else:
            cell = np.multiply(cell, color)
        image[row.pxtop:row.pxbottom, row.pxleft:row.pxright] = np.minimum(cell, [255, 255, 255]).astype(int)

    for row in grid.itertuples():
        if np.isnan(row.max_depth) or row.max_depth>args.mindepth:
            highlight_cell(color, [0.2, 0.2, 0.2], row)
        elif row.max_depth < args.mindepth and row.mean_ndvi > filter_ndvi:
            highlight_cell(color, [3.0, 0, 0], row)

    plt.imsave(arr=color, fname=os.path.join(landsat_dir_name, os.path.dirname(landsat_dir_name)+".jpg"))

print("Saving data...")
sea = grid[grid["max_depth"] < args.mindepth].copy()
kelp = sea[sea["mean_ndvi"] > filter_ndvi].copy()

kelp["lat"] = (kelp["top"] - kelp["bottom"]) / 2.0 + kelp["bottom"]
kelp["lon"] = (kelp["right"] - kelp["left"]) / 2.0 + kelp["left"]
final_kelp=kelp[['lat','lon', 'mean_ndvi']]
final_kelp.columns = ["lat", "lon", "ndvi"]
final_kelp.to_csv(args.output, index=False)

print("Done.")
