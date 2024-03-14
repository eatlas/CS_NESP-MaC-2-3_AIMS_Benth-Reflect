"""
Australian Institute of Marine Science
Eric Lawrey

This script estimates the benthic reflectance from satellite imagery and bathymetry.
The satellite image brightness is scaled pixel by pixel between the estimated brightness
of a low and high benthic reflectance. The pixel level estimates are determined from 
estimating the expected brightness for high and low benthic reflectance using the
bathymetry combined with a model of the depth verse brightness and benthic reflectance.
This model is determnined by fitting a curve to observed sample points of locations
classified manually as high or low benthic reflectance.  

Before this script is run the bathymetry and satellite imagery datasets should be downloaded. 
This can be achieved by running the 01-download-src-data.py script, or using the links in the script
to manually download these files.

This script crops and aligns the bathymetry to the satellite imagery. This is then 
clipped into each of the swath polygons in the swath_analysis_areas_file shapefile. 

We do this because there are slight bright and dark bands in the Sentinel 2 imagery due to
the staggered detector configuration of the satellite. These slight differences will result
in a change in the depth vs brightness across each detector in the swath. We therefore
cut up the image into each detector swath and perform separate analysis on each part.

This script extracts matching depth and satellite image values for the locations specified
in the new-data/Depth-Reflect-Sampling-Points.shp file. 

It then models the depth vs brightness for each area. We then synthesis how bright we would expect
the area to be from the bathymetry, for both low and high reflectance. The satellite imagery
brightness is then scaled between these low and high limits to estimate the reflectance.

The analysis for each area and trial is saved in its own working folder. This contains intermediate
calculations such as clipped and aligned satellite and bathymetry, the extracted matching 
bathymetry and brightness values, and synthesised expected brightness for low and high reflectance.


Data dictionary of output:
output/*/02B-Depth_Reflect-class_S2-Bright.csv
Latitude: Location of point sample
Longitude: Location of point sample
ID: Sequential counter of point sample
Reflect: Substrate brightness, either 'High' or 'Low'
SWATH_SEG: Integer 1 or 2, corresponding to two separate areas to repeat the modelling over.
Depth_m: Bathymetry of the point sample in metres
S2_R1_B1: Sentinel 2 image brightness band 1 (UV)
S2_R1_B2: Sentinel 2 image brightness band 2 (Blue)
S2_R1_B3: Sentinel 2 image brightness band 3 (Green)
S2_R1_B4: Sentinel 2 image brightness band 4 (Red)

output/03C-benthic-reflect_SEG_{swath}.tif
Estimated benthic reflectance scaled from 1 - 255. 0 is reserved for no data.
"""

import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
import pandas as pd
import numpy as np
import os
from shapely.geometry import mapping
import time
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import box
from scipy.ndimage import gaussian_filter, uniform_filter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import plotly.express as px
import sys

# Resampling to apply to the bathymetry when aligning it to the satellite imagery.
# Resampling.nearest
# Resampling.bilinear
RESAMPLING = Resampling.bilinear



# Use to save intermediate calculations that can be deleted at the end 
TEMP_PATH = 'temp'

# Base path for all the input datasets
DATA_CACHE = 'working/data-cache'

# Points in the imagery to analyse the relationship between depth in brightness. This shapefile
# should have an attribute Reflect, with a value of 'High' for sandy areas, and 'Low' for locations
# with a dark surface. The 'Low' reflectance can be vegetation or reefal areas. Generally any area that 
# is obviously not sand. The sampling points were chosen in locations where the classification of
# reflectance was uniform for at least 30 m in every direction.
DEPTH_REFLECTANCE_SAMPLING_POINTS_FILE = 'new-data/Depth-Reflectance-Sampling-Points.shp'



# This section defines the parameters of each analysis. Each analysis corresponds to
# a combination of Sentinel 2 imagery and bathymetry dataset.
# The number of values should match for 
# - BATHYMETRY_FILES, 
# - SENTINEL2_IMG_TILE_FILES, 
# - SWATH_AREA_ID_LISTS, 
# - MODEL_PARAM_ADJ_Lbj_K_R_RATIO,
# - FITTED_HIGH_REFLECTANCE
# - FITTED_LOW_REFLECTANCE,
# - OUTPUT_PATHS

# Input Sentinel 2 imagery to perform the depth / brightness relationship with. 
SENTINEL2_IMG_TILE_FILES = [
    f'{DATA_CACHE}/S2_Benth-Ref/Wld_AIMS_Marine-sat-img_S2_Raw-B1-B4_55KFA.tif',
    f'{DATA_CACHE}/S2_Benth-Ref/Wld_AIMS_Marine-sat-img_S2_20200818_Raw-B1-B4_55KFA.tif',
    f'{DATA_CACHE}/S2_Benth-Ref/Wld_AIMS_Marine-sat-img_S2_Raw-B1-B4_55KEB.tif',
    f'{DATA_CACHE}/S2_Benth-Ref/Wld_AIMS_Marine-sat-img_S2_NoSGC_Raw-B1-B4_55KFA.tif',
    f'{DATA_CACHE}/S2_Benth-Ref/Wld_AIMS_Marine-sat-img_S2_Raw-B1-B4_55KFA.tif',
    f'{DATA_CACHE}/S2_Benth-Ref/Wld_AIMS_Marine-sat-img_S2_Raw-B1-B4_56KLF.tif',
    f'{DATA_CACHE}/S2_Benth-Ref/Wld_AIMS_Marine-sat-img_S2_Raw-B1-B4_55KEV.tif'
    ]
    

# List of bathymetry files to use for each analysis. 
gbr30_B_30m_file = f'{DATA_CACHE}/GBR_GA_Great-Barrier-Reef-Bathy-30m_2020/Great_Barrier_Reef_B_2020_30m_MSL_cog.tif'
gbr100_100m_file = f'{DATA_CACHE}/GBR_GA_Great-Barrier-Reef-Bathy-100m_2020/Great_Barrier_Reef_2020_100m_MSL_cog.tif'
BATHYMETRY_FILES = [
    gbr30_B_30m_file,
    gbr30_B_30m_file,
    gbr30_B_30m_file,
    gbr30_B_30m_file,
    gbr100_100m_file,
    gbr100_100m_file,
    gbr30_B_30m_file
    ]


    



# Sentinel 2 has slight changes in brightness across the swath due to the staggered detector
# configuration. https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/msi-instrument
# This causes segments of the image, at an angle of about 13 deg off the vertical, to appear slightly
# lighter or darker than other segments. Since this is uncorrected in the imagery we segment each
# image into areas that have a single uniform brightness. We manually create a shapefile that splits 
# the Sentinel 2 imagery into detector swaths to improve brightness uniformity. This shapefile has an
# attribute 'SWATH_SEG' that has values 1 to 9 to indicate each region to perform the analysis. This
# shapefile is shared across all analyses.
# These areas are used to clip the bathymetry and satellite imagery.
SWATH_ANALYSIS_AREAS_FILE = 'new-data/Swath-analysis-areas-Poly.shp'

# List of the swath shapes to perform the analsysis on.
SWATH_AREA_ID_LISTS = [
    [1,2],
    [1,2],
    [3],
    [1,2],
    [1,2],
    [4, 5, 6, 7],
    [8,9]]

# To fit the brightness vs depth curves we need to specify an initial guess for the model 
# parameters and a valid range of values. The initial guess is based on model parameters 
# obtained for North Flinders Reef where the data is the best in the Coral Sea. 
# This specifies the minimum and maximum range as a ratio of the initial guess for each
# model run. This can be used to constrain a model parameter, which is useful if there are
# very few valid data points. If the values were 1.0, 1.0, 1.0 then the model would be
# constrained to the one developed for North Flinders reef.
MODEL_PARAM_ADJ_Lbj_K_Lwj_RATIO = [
    [1.2, 1.2, 1.2],
    [1.2, 1.2, 1.2],
    [1.2, 1.2, 1.2],
    [1.2, 1.2, 1.2],
    [1.2, 1.2, 1.2],
    [1.2, 1.2, 1.2],
    [1.2, 1.2, 1.2]
]

# The fitted curve for the high and low reflectance don't correspond to a reflectance
# of 1 and 0, as the sampling points don't correspond to black and white surfaces
# but reef rock, halimeda and sand. 

# Average reflectance of sampled points in the modelled area. Typically this
# should correspond to the reflectance of sand.
FITTED_HIGH_REFLECTANCE = [
    0.8,
    0.8,
    0.8,
    0.8,
    0.8,
    0.8,
    0.8
    ]
# Average reflectance of the low reflectance points sampled (reef rock / halimeda)
FITTED_LOW_REFLECTANCE = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]    

# This is the root path where temporary and intermediate calculation files are saved.
WORKING_BASE_PATH = 'working/depth-reflect'

# This is the root path for files that should be made available as part of the
# public datasets matching this analysis. This includes the reflectance,
# depth brightness extraction, model fits parameters and plots.
OUTPUT_BASE_PATH = 'output'

# Names of the folders to save results of each analysis in the working and output folders.
ANALYSIS_FOLDERS = [
    '55KFA-8',
    '55KFA-1',
    '55KEB',
    '55KFA-8-NoSGC',
    '55KFA-8-gbr100',
    '56KLF-7-gbr100',
    '55KEV'
    ]

# Clip reflectance predictions for areas deeper then this threshold
BATHY_THRESHOLD = -70

# Range of bathymetries to plot the curves over
MIN_PRED_BATHY = {
    "B1": -65,
    "B2": -65,
    "B3": -40,
    "B4": -15,
}
MAX_PRED_BATHY = {
    "B1": -1,
    "B2": -1,
    "B3": -1,
    "B4": -1,
}

# Radius of the gaussian blurring applied to the reflectance prediction to reduce noise
REFLECTANCE_FILTER_SIGMA = 1.5

# The reflectance range is used to scale the contrast based on depth.
# Deep areas have a small range between high and low reflectance, 
# shallow areas have a high contrast. This is in units of the
# original satellite, so black to white is about 8000.
# This sets the maximum contrast enhancement to apply to estimate
# the reflectance. 
# Slight errors in the modelling of the brightness / depth relationship
# and slight drifts in tonal values across the scene get magnified too
# much if we don't limit the maximum contrast enhancement by limiting the
# reflectance range.
REFLECTANCE_RANGE_LOWER_LIMIT = 25   # Units tonal levels of raw sentinel image

BANDS = ['B1', 'B2', 'B3', 'B4']

BATHY_PATH_TEMPLATE = '{working_path}02A_bathy_aligned_bathy_SEG_{swath}.tif'
SATELLITE_PATH_TEMPLATE = '{working_path}02A_satellite_imagery_SEG_{swath}.tif'
MODEL_PARAMS_PATH_TEMPLATE = '{output_path}03A-brightness-vs-depth_models.csv'
# Path to save the extracted bathymetry, brightness and reflectance.
DEPTH_REFLECTANCE_CSV_PATH_TEMPLATE = '{output_path}02B-Depth_Reflect-class_S2-Bright.csv'
SYNTH_PATH_TEMPLATE = '{working_path}03B-brightness-from-depth-{reflectance}_SEG_{swath}.tif'
PLOT_PATH_TEMPLATE = '{output_path}03B-bright-vs-depth_reflect-{reflectance}_SEG-{swath}.png'
BAND_PLOT_PATH_TEMPLATE = '{output_path}03B-bright-vs-depth_{band}.png'
REFLECTANCE_RANGE_IMAGE_PATH_TEMPLATE = '{working_path}03C-reflectance-range_SEG_{swath}.tif'
NORMALIZED_REFLECTANCE_IMAGE_PATH_TEMPLATE = '{output_path}03C-benthic-reflect_SEG_{swath}.tif'
DONE_PATH_TEMPLATE = '{working_path}04-done-flag.txt'


# Save the intermediate range between the predicted high and low reflectance
SAVE_REFLECTANCE_RANGE_IMAGE = False

# Interactive graphs for inspecting outliers
INTERACTIVE_GRAPHS = False

# Best estimates from 8 image composite from
# Flinder Reef (55KFA-8). These can be used when there
# is insufficent data to determine for a specific image

# Lwj is the Rayleigh scattered light
INITIAL_Lwj = {
    'B1': 1206,
    'B2': 767,
    'B3': 357,
    'B4': 124
}

# Kj is the attenuation rate through water
INITIAL_Kj = {
    'B1':0.038,
    'B2': 0.045,
    'B3': 0.084,
    'B4': 0.35
}

# Lbj is the scaling of the brightness curve above the background
# rayleigh scattering, i.e. the amount of benthic relected light.  
INITIAL_Lbj = {
    'High': {
        'B1': 2680,
        'B2': 3052,
        'B3': 3366,
        'B4': 12350
        },
    'Low': {
        'B1': 1536,
        'B2': 1269,
        'B3': 1282,
        'B4': 792
        }
    }

# From https://pygis.io/docs/e_raster_resample.html
def reproj_match(infile, match, outfile, resampling):
    """Reproject a file to match the shape and projection of existing raster. 
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacKjs input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 0,
                           "compress": 'lzw'})
                            #"dtype": 'int16'})
        #print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling)
                
def clip_to_swaths_and_save(raster_path, swath_polygons, working_path, working_path_template, swath_attr, output_dtype='float32'):
    with rasterio.open(raster_path) as src:
        nodata_value = src.nodata if src.nodata is not None else -9999  # Use existing nodata or default
        for index, polygon in swath_polygons.iterrows():
            out_image, out_transform = mask(src, [polygon.geometry], crop=True, nodata=nodata_value)
            out_meta = src.meta.copy()

            # Handle data type conversion and ensure nodata value is set correctly
            if output_dtype == 'int16':
                out_image = np.round(out_image).astype('int16')
                nodata_value = -9999  # Ensure nodata value is appropriate for int16
            elif output_dtype == 'float32':
                out_image = out_image.astype('float32')

            # Update metadata for the output file
            out_meta.update(dtype=output_dtype, nodata=nodata_value, driver="GTiff",
                            height=out_image.shape[1], width=out_image.shape[2], 
                            transform=out_transform, compress='lzw')

            output_file = working_path_template.format(working_path = working_path, swath = polygon[swath_attr])
            print(f"Saving clipped raster to {output_file}")
            
            # Set nodata value directly in rasterio.open call
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(out_image)

# Function to clip raster to the extent of another raster and save the output
def clip_to_extent(infile, extent_file, outfile):
    with rasterio.open(infile) as src:
        with rasterio.open(extent_file) as extent_src:
            # Create a bounding box polygon from the extent raster
            bbox = box(*extent_src.bounds)
            # Clip the source raster using the bounding box
            out_image, out_transform = rasterio.mask.mask(src, [bbox], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": 'lzw'
            })
            print(f"Clipping {os.path.basename(infile)} to the extent of {os.path.basename(extent_file)}")
            with rasterio.open(outfile, "w", **out_meta) as dest:
                dest.write(out_image)
                
# Check if the required files exist
def check_files_exist(*files):
    for file in files:
        if not os.path.exists(file):
            msg = f"{file} not found. Please ensure the file exists."
            raise FileNotFoundError(msg)

# Apply a 3x3 averaging kernel filter to smooth the imagery
def apply_averaging_kernel(image, band=1):
    with rasterio.open(image) as src:
        data = src.read(band)
        smoothed_data = uniform_filter(data, size=3, mode='nearest')
        affine_transform = src.transform.copy()  # Copy the profile while the dataset is open
    return smoothed_data, profile


def extract_raster_values(points_df, raster_file, prefix="", filter_type=None, filter_size=1):
    """
    Extract raster values for given points from all bands of an input raster file, with an option to apply filtering.

    Parameters:
    - points_df: A GeoDataFrame containing points of interest with their geometries.
    - raster_file: Filename of the image to extract data from.
    - prefix: Optional prefix to apply to band names in the output DataFrame. If there is only 1 band then
              use this as the whole name.
    - filter_type: Optional filter type to apply ('gaussian' for Gaussian filter, 'moving_average' for moving average).
    - filter_size: Sigma radius of the 'gaussian' filter, or kernal size for 'moving_average'. Default is 1.

    Returns:
    - A DataFrame with one column per band and one row per point, containing the extracted values.
    """
    with rasterio.open(raster_file) as src:
        affine_transform = src.transform
        if src.count == 1:
            band_names = [prefix]
        else:
            # 
            band_names = [prefix + src.descriptions[i] if src.descriptions[i] else prefix + str(i+1) for i in range(src.count)]
        print(f"{band_names=}")
        values_df = pd.DataFrame(columns=band_names, index=points_df.index)
        
        raster_data = src.read()

        # Apply filtering if requested
        if filter_type == 'gaussian':
            raster_data = gaussian_filter(raster_data, sigma=(0, filter_size, filter_size), mode='nearest')
        elif filter_type == 'moving_average':
            raster_data = uniform_filter(raster_data, size=(1, filter_size, filter_size), mode='nearest')
        
        for index, point in points_df.iterrows():
            px, py = ~affine_transform * (point.geometry.x, point.geometry.y)
            px, py = int(px), int(py)
            
            for band_index, band_name in enumerate(band_names):
                if (0 <= px < raster_data.shape[2]) and (0 <= py < raster_data.shape[1]):
                    value = raster_data[band_index, py, px]
                    values_df.at[index, band_name] = value
                else:
                    values_df.at[index, band_name] = np.nan
        for col in values_df.columns:
            values_df[col] = pd.to_numeric(values_df[col], errors='coerce')

        return values_df

def pre_data(bathymetry_file, sentinel2_img_tile_file, swath_area_id_list, swath_analysis_areas_file, 
    depth_reflect_sampling_points_file, working_path, output_path, resampling, temp_path):
    # Process starts here
    print("Starting process...")
    
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        print(f"Creating output path: {output_path}")
        os.makedirs(output_path)
        
    if not os.path.exists(working_path):
        print(f"Creating output path: {working_path}")
        os.makedirs(working_path)
        
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    # A) -------------- Crop, align and clip rasters ---------------
    # Fully resolve the path relative to this directory. This is needed because the path is close to
    # the maximum Windows path length.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sentinel2_img_file = os.path.normpath(os.path.join(script_dir, sentinel2_img_tile_file))

    # Preliminary checks
    check_files_exist(bathymetry_file, sentinel2_img_file, depth_reflect_sampling_points_file, swath_analysis_areas_file)

    # Clip the bathymetry to the extent of the Sentinel 2 imagery
    print("Clipping bathymetry to imagery.")
    clipped_bathy_path = os.path.join(temp_path, 'clipped_bathy.tif')
    clip_to_extent(bathymetry_file, sentinel2_img_file, clipped_bathy_path)

    # Co-align the clipped bathymetry with the Sentinel 2 imagery
    print("Aligning bathymetry to imagery.")
    aligned_bathy_path = os.path.join(temp_path, 'aligned_bathy.tif')
    reproj_match(clipped_bathy_path, sentinel2_img_file, aligned_bathy_path, resampling)
    
    
    # Read the swath analysis areas shapefile
    print("Loading swath analysis areas.")
    swath_polygons = gpd.read_file(swath_analysis_areas_file)
    
    # Filter the GeoDataFrame
    filtered_swath_polygons = swath_polygons[swath_polygons['SWATH_SEG'].isin(swath_area_id_list)]


    # Clip and save both datasets for each swath
    print("Clipping bathymetry to swath paths.")
    clip_to_swaths_and_save(aligned_bathy_path, filtered_swath_polygons, working_path, BATHY_PATH_TEMPLATE, 'SWATH_SEG')
    
    print("Clipping imagery to swath paths.")
    clip_to_swaths_and_save(sentinel2_img_file, filtered_swath_polygons, working_path, SATELLITE_PATH_TEMPLATE, 'SWATH_SEG', output_dtype='int16')

    

    # ------------- Extract match Depth and Reflectance -----------------
    print("Loading sampling points shapefile")
    points_gdf = gpd.read_file(DEPTH_REFLECTANCE_SAMPLING_POINTS_FILE)

    print("Joining Swath Segments")
    joined_points_gdf = gpd.sjoin(points_gdf, filtered_swath_polygons[['geometry', 'SWATH_SEG']], how="left", predicate='intersects')

    print("Extracting points from bathymetry")
    bathymetry_values_df = extract_raster_values(joined_points_gdf, aligned_bathy_path, 'Depth_m')

    print("Extracting points from sentinel 2")
    # Adding gaussian filtering prior to extraction doesn't make an improvement, its
    # effect is relatively minimal.
    s2_values_df = extract_raster_values(joined_points_gdf, sentinel2_img_file, '')
        
    # Extract Latitude and Longitude from the geometry
    joined_points_gdf['Latitude'] = joined_points_gdf.geometry.y
    joined_points_gdf['Longitude'] = joined_points_gdf.geometry.x

    # Combine the geometries and additional attributes from 'joined_points_gdf' with the raster values
    combined_df = pd.concat([
        joined_points_gdf[['Latitude', 'Longitude', 'ID', 'Reflect', 'SWATH_SEG']],
        bathymetry_values_df,
        s2_values_df
    ], axis=1)
    
    # Check for NaN values in the 'ID' column
    nan_count = combined_df['ID'].isna().sum()
    if nan_count > 0:
        raise ValueError(f"Error: There are {nan_count} NaN values in the 'ID' column.")


    # Filtering to include only rows where SwathSeg is not null/empty
    # After filtering the DataFrame, explicitly create a copy to avoid SettingWithCopyWarning
    filtered_combined_df = combined_df.dropna(subset=['SWATH_SEG', 'Depth_m', 'B1', 'B2', 'B3', 'B4']).copy()

    # Convert 'ID' and 'SWATH_SEG' to integers
    # If you expect no missing values after the filtering or if your DataFrame supports it, you can directly convert to integers.
    filtered_combined_df['ID'] = filtered_combined_df['ID'].astype(int)
    filtered_combined_df['SWATH_SEG'] = filtered_combined_df['SWATH_SEG'].astype(int)
    
    output_csv_path = DEPTH_REFLECTANCE_CSV_PATH_TEMPLATE.format(output_path = output_path)
    # Extract the directory path from the output_csv_path
    output_directory = os.path.dirname(output_csv_path)

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Export to CSV
    filtered_combined_df.to_csv(output_csv_path, index=False)
    print(f"Exported extracted values to {output_csv_path}")
    
    # Convert 'ID' and 'SWATH_SEG' to integers
    # If you expect no missing values after the filtering or if your DataFrame supports it, you can directly convert to integers.
    filtered_combined_df['ID'] = filtered_combined_df['ID'].astype(int)
    filtered_combined_df['SWATH_SEG'] = filtered_combined_df['SWATH_SEG'].astype(int)
    
    # Clip
    
    # Clean up the temp files.
    os.remove(clipped_bathy_path)
    os.remove(aligned_bathy_path)
    return(filtered_combined_df)
    
    
    


# A) ----------------- Fit model to brightness vs depth -----------------

# The reflected light is determined by the attenuation coefficient k (fraction of light
# lost in each metre of water), Lwj (amount of scattered reflected light from deep water),
# and Lbj is the relative brightness of the wet substrate with no water cover.
# Jupp, D. L. B., 1988. Background and extensions to Depth of Penetration (DOP) 
# mapping in shallow coastal waters. Symposium on Remote Sensingof the Coastal Zone. 
# Gold Coast, Queensland, Session 4, Paper 2
#
# Lbj - Radiance of the wet substrate with no water cover (depth = 0) (IO)
# Lwj - Radiance of deep water at frequency j (R)
# Kj - Depth averaged attenuation coefficient of the water. This is the fraction of 
# light lost in each metre of water. The value of Kj varies with wavelength as well
# as composition of the water, with blue light penetrating deepest (smallest value 
# of Kj).

def brightness_verses_depth_model_jupp_1988(z, Lbj, Kj, Lwj):
    # Effective 'transmittance' of the water at the frequency being modelled.
    # twj varies between 0 and 1 and represents the fraction of light lost
    # travelling down and up through the water.
    twj = np.exp(-2 * Kj * (-z))
    #return Lbj * twj + Lwj
    return Lbj * twj + Lwj*(1-twj)

def calculate_Lwj_values(df, bands):
    # Filter samples that are deeper than -60 m
    deep_water_samples = df[df['Depth_m'] < -60]
    
    # Check if there are enough deep water samples
    if not deep_water_samples.empty:
        # If there are, calculate the mean for each band
        return {band: deep_water_samples[band].mean() for band in bands}
    else:
        # If not, sort the DataFrame by depth (ascending=False to get deepest first)
        deepest_samples = df.sort_values('Depth_m', ascending=False).head(10)
        # Then calculate the mean for each band using these 10 samples
        return {band: deepest_samples[band].mean() for band in bands}


    
    
def fit_models(df, bands, output_path, param_adj_ratio_Lbj_Kj_Lwj =[1.5, 1.5, 1.5]):

    #if not os.path.exists(model_params_file):
    # Calculate the initial Lwj values
    Lwj_values = calculate_Lwj_values(df, bands)
    print(f'{Lwj_values=}')

    # Fit the model and generate the plot
    model_params = []

    # Best estimates from 8 image composite from
    # Flinder Reef (55KFA-8). These can be used when there
    # is insufficent data to determine for a specific image
   
    
    # Determine how much slack the model has to tune parameters away from 
    # Best estimate from Flinder Reef
    par_Lbj = param_adj_ratio_Lbj_Kj_Lwj[0]
    par_Kj = param_adj_ratio_Lbj_Kj_Lwj[1]
    par_Lwj = param_adj_ratio_Lbj_Kj_Lwj[2]
    
    for band in bands:
        for swath in df['SWATH_SEG'].unique():
            print(f'{band=}, {swath=}')
            swath = int(swath)
            # Estimate Lwj values for low and high reflectance. The Lwj value should be
            # identical for both sets of data. In some cases the Lwj value for
            # low reflectance will be higher than the high reflectance, resulting in
            # an inversion of the contrast enhancement. To prevent this we
            # estimate Lwj for both 'High' and 'Low' and take the average.
            # This works assuming that both the 'High' and 'Low' are balanced
            # and have an equal number of deep points. 
            # We then use this fixed value to refit the model.
            maxfev = 10000
            
            high_Lwj = INITIAL_Lwj[band]*par_Lwj
            low_Lwj = INITIAL_Lwj[band]/par_Lwj
            initial_Lwj = INITIAL_Lwj[band]
            
            # Merge the estimate of Lwj from the fitted high and low reflectance.
            # This helps ensure that the low is not above the high estimate.
            COMBINE_HIGH_LOW_Lwj = True
            if COMBINE_HIGH_LOW_Lwj:
            
                Lwj_values = []
                for reflectance in df['Reflect'].unique(): 
                    p0=[INITIAL_Lbj[reflectance][band], INITIAL_Kj[band], initial_Lwj]
                    initial_bounds = (
                            [INITIAL_Lbj[reflectance][band]/par_Lbj, INITIAL_Kj[band]/par_Kj, low_Lwj], 
                            [INITIAL_Lbj[reflectance][band]*par_Lbj, INITIAL_Kj[band]*par_Kj, high_Lwj])
                    
                    filtered_data = df[(df['Reflect'] == reflectance) & (df['SWATH_SEG'] == swath)]
                    popt, pcov = curve_fit(
                        brightness_verses_depth_model_jupp_1988,
                        filtered_data['Depth_m'],
                        filtered_data[band],
                        p0=p0,  # initial guess
                        bounds=initial_bounds,  # bounds
                        maxfev=maxfev,
                        method='trf'    # For constrained problem (bounds)
                    )
                    
                    Lwj_values.append(popt[2])
                # Determine the combined estimate for Lwj
                high_low_Lwj = np.mean(np.array(Lwj_values))
                print(f"{high_low_Lwj=}")
                # The upper bound can't be the same as the lower bound and
                # so we add a very small gap. If eps is too small then the
                # model fit might fail.
                eps = 5   # Units are in integer brightness from 0 - 10000
                high_Lwj = high_low_Lwj+eps
                low_Lwj = high_low_Lwj-eps
                initial_Lwj = high_low_Lwj
                


            # If combining high and low Lwj then reperform the modelling, but with a 
            # fixed value of Lwj, otherwise just start with the initial estimates
            for reflectance in df['Reflect'].unique():
                p0=[INITIAL_Lbj[reflectance][band], INITIAL_Kj[band], initial_Lwj]
                
                initial_bounds = (
                        [INITIAL_Lbj[reflectance][band]/par_Lbj, INITIAL_Kj[band]/par_Kj, low_Lwj], 
                        [INITIAL_Lbj[reflectance][band]*par_Lbj, INITIAL_Kj[band]*par_Kj, high_Lwj])
                
                filtered_data = df[(df['Reflect'] == reflectance) & (df['SWATH_SEG'] == swath)]
                popt, pcov = curve_fit(
                    brightness_verses_depth_model_jupp_1988,
                    filtered_data['Depth_m'],
                    filtered_data[band],
                    p0=p0,  # initial guess
                    bounds=initial_bounds,  # bounds
                    maxfev=maxfev,
                    method='trf'    # For constrained problem (bounds)
                )
            
                # Determine the model fit.
                y_true = filtered_data[band]
                y_pred = brightness_verses_depth_model_jupp_1988(filtered_data['Depth_m'], *popt)
                r2 = r2_score(y_true, y_pred)
                
                print(f"{reflectance=}, estimated model parameters: {popt}, R2: {r2}") 
                
                model_params.append({'Reflect': reflectance, 'SWATH_SEG': swath, 
                    'Band': band, 'Lbj': popt[0], 'Kj': popt[1], 'Lwj': popt[2], 'R2': r2})

    # Convert the model parameters to a DataFrame and save
    model_params_df = pd.DataFrame(model_params)
    model_params_file = MODEL_PARAMS_PATH_TEMPLATE.format(output_path = output_path)
    model_params_df.to_csv(model_params_file, index=False)
    return(model_params_df)
        
def plot_data_and_models(df, bands, output_path, model_params_df):
    # Now generate plots using the prediction data
    # This should be automated, but it isn't. Each new segement needs its colour specified.
    colors = {
    'High': {'1': 'cyan', '2': 'green', '3': 'orange', '4':'cyan', '5': 'green', '6': 'orange', '7':'grey', '8':'cyan', '9': 'green',}, 
    'Low': {'1': 'blue', '2': 'olive', '3':'brown','4': 'blue', '5': 'olive', '6':'brown','7': 'black', '8': 'blue', '9': 'olive'}
    }
    
    # Perform the model fitting and save the prediction data
    prediction_data = {}
    
    for band in bands:
        prediction_data[band] = {}
        fig, ax = plt.subplots(figsize=(10, 8))
        # Plot the observed data points and prediction lines
        for reflectance in df['Reflect'].unique():
            prediction_data[band][reflectance] = {}
            for swath in df['SWATH_SEG'].unique():
                swath = int(swath)
                filtered_data = df[(df['Reflect'] == reflectance) & (df['SWATH_SEG'] == swath)]
                print(f'Plotting {band=}, {reflectance=}, {swath=}')
                # Save the prediction data
                #z_fit = np.linspace(filtered_data['Depth_m'].min(), filtered_data['Depth_m'].max(), 100)
                z_fit = np.linspace(MIN_PRED_BATHY[band], MAX_PRED_BATHY[band], 200)
                
                band_params = model_params_df[(model_params_df['Band'] == band) &
                                               (model_params_df['Reflect'] == reflectance) &
                                               (model_params_df['SWATH_SEG'] == swath)]
                    
                model_params_row = band_params.iloc[0]
                Lbj = model_params_row['Lbj']
                Kj = model_params_row['Kj']
                Lwj = model_params_row['Lwj']
                L_fit = brightness_verses_depth_model_jupp_1988(z_fit, Lbj, Kj, Lwj)
                prediction_data[band][reflectance][swath] = (z_fit, L_fit)
                
                # Plot the observed data
                ax.scatter(filtered_data['Depth_m'], filtered_data[band], 
                    color=colors[reflectance][str(swath)], s=10, label=f'Observed: {reflectance} Swath {swath}')
                
                # Get the prediction data
                z_fit, L_fit = prediction_data[band][reflectance][swath]
                # Plot the prediction line
                ax.plot(z_fit, L_fit, color=colors[reflectance][str(swath)], label=f'Predicted: {reflectance} Swath {swath}')
        
        # Labeling the plot
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Brightness')
        ax.set_title(f'Brightness vs Depth for {band}')
        ax.legend()
        
        # Limit the x-axis range
        ax.set_xlim(MIN_PRED_BATHY[band], MAX_PRED_BATHY[band])
        
        # Save the plot
        plot_filename = BAND_PLOT_PATH_TEMPLATE.format(output_path=output_path, band=band)
        plt.savefig(plot_filename)
        plt.close()

    # Plot the difference between high and low reflectance values.
    for band in bands:
        fig, ax = plt.subplots(figsize=(10, 8))
        for swath in df['SWATH_SEG'].unique():
            swath = int(swath)

            # Assuming you have a way to calculate these differences
            # For example, high_reflectance - low_reflectance for each depth value
            z_fit = prediction_data[band]['High'][swath][0]  # Assuming z_fit is the same for High and Low
            high_reflectance_fit = prediction_data[band]['High'][swath][1]
            low_reflectance_fit = prediction_data[band]['Low'][swath][1]
            difference_fit = high_reflectance_fit - low_reflectance_fit

            # Plot the difference
            ax.plot(z_fit, difference_fit, label=f'Swath {swath}', color=colors['High'][str(swath)])  # Using High color for simplicity

        ax.set_yscale('log')

        ax.set_ylim(10, 1500)
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('High - Low Reflectance Difference')
        ax.set_title(f'Reflectance Difference vs Depth for {band}')
        ax.legend()

        # Save the plot
        plot_filename = f'{output_path}reflectance_difference_{band}.png'
        plt.savefig(plot_filename)
        plt.close()

    # Assuming 'df' is your original DataFrame
    df_long = df.melt(id_vars=['Latitude', 'Longitude', 'ID', 'Reflect', 'SWATH_SEG', 'Depth_m'],
                      value_vars=bands,
                      var_name='Band', value_name='Brightness')

    # --------------------- Interactive plot to investigate outliers --------------
    if INTERACTIVE_GRAPHS:
        # Now use the restructured DataFrame 'df_long' for your Plotly Express plot
        for reflectance in df_long['Reflect'].unique():
            for swath in df_long['SWATH_SEG'].unique():
                # Filter data for the current reflectance and swath
                filtered_data = df_long[(df_long['Reflect'] == reflectance) & (df_long['SWATH_SEG'] == swath)]

                # Generate an interactive Plotly plot
                fig = px.scatter(filtered_data, x='Depth_m', y='Brightness', color='Band',
                                 title=f'Brightness vs Depth - Reflectance: {reflectance}, Swath: {swath}',
                                 labels={'Depth_m': 'Depth (m)', 'Brightness': 'Brightness'},
                                 hover_data=['ID'])
                fig.update_traces(mode='markers')
                fig.show()

                # Wait for user input to move on to the next plot
                #input("Press Enter to continue to the next plot...")

def synthesise_imagery_from_bathy(df, bands, working_path, output_path, model_params_df):
    # B) -------------- Synthesise expected imagery from bathymetry ---------------
    print("Predicting the brightness from the depth")
    # Loop through each swath and reflectance to generate the images
    for reflectance in df['Reflect'].unique():
        for swath in df['SWATH_SEG'].unique():  # Assuming swath segments are numbered 1 and 2
            swath = int(swath)
            print(f'Processing {reflectance = }, {swath = }')
            synth_path = SYNTH_PATH_TEMPLATE.format(working_path = working_path, reflectance = reflectance, swath = swath)

            
            # Load the bathymetry file for the current swath
            bathy_path = BATHY_PATH_TEMPLATE.format(working_path = working_path, swath=swath)
            with rasterio.open(bathy_path) as bathy_src:
                bathy_data = bathy_src.read(1, masked=True)  # Read with masking based on no-data value
                no_data_value = bathy_src.nodata  # Read the no-data value from the source
                print(f'{no_data_value=}')
                
                
                # Initialize an empty array to hold the synthesized image data
                synthesized_image = np.zeros((4, bathy_src.height, bathy_src.width), dtype=np.float32)

                # Loop through each band and predict brightness based on the bathymetry
                for band_idx, band in enumerate(bands, start=1):
                    # Retrieve the model parameters for the current band, swath, and reflectance
                    
                    band_params = model_params_df[(model_params_df['Band'] == band) &
                                               (model_params_df['Reflect'] == reflectance) &
                                               (model_params_df['SWATH_SEG'] == swath)]
                    
                    model_params_row = band_params.iloc[0]
                    Lbj = model_params_row['Lbj']
                    Kj = model_params_row['Kj']
                    Lwj = model_params_row['Lwj']
                    # Predict the brightness for the current band
                    predicted_brightness = brightness_verses_depth_model_jupp_1988(bathy_data, Lbj, Kj, Lwj)
                    synthesized_image[band_idx - 1] = np.where(bathy_data.mask, bathy_src.nodata, predicted_brightness)
                    
                # Write the synthesized image to a new .tif file

                with rasterio.open(
                    synth_path, 'w',
                    driver='GTiff',
                    height=bathy_src.height,
                    width=bathy_src.width,
                    count=4,
                    dtype=synthesized_image.dtype,
                    crs=bathy_src.crs,
                    transform=bathy_src.transform,
                    nodata=0,  # Set the no-data value for the output
                    compress='lzw'
                ) as dst:
                    dst.write(synthesized_image)

# C) ---------------------- Estimate reflectance ------------------------
# In this section we try to estimate the reflectance from the satellite imagery and the synthesised
# high and low reflectance from the bathymetry. The main idea is that if the image is about the same
# brightness as high reflectance image estimated from bathymetry then those parts of the image are
# high reflectance. Areas in the image that are darker than expected have a lower reflectance.
# The main challenge is that the contrast between high and low reflectance gets lower in deeper areas
# and so we need to compensate for this. This introduces a lot of noise in the estimates.
def process_imagery(swath, model_params, working_path, FITTED_HIGH_REFLECTANCE, FITTED_LOW_REFLECTANCE, bathymetry_threshold=-70):
    # Load the synthesized images for high and low reflectance
    high_reflectance_image_path = SYNTH_PATH_TEMPLATE.format(working_path = working_path, reflectance = 'High', swath = swath)
    low_reflectance_image_path = SYNTH_PATH_TEMPLATE.format(working_path = working_path, reflectance = 'Low', swath = swath)
    satellite_image_path = SATELLITE_PATH_TEMPLATE.format(working_path = working_path, swath = swath)
    bathy_path = BATHY_PATH_TEMPLATE.format(working_path=working_path, swath=swath)

    with rasterio.open(high_reflectance_image_path) as high_src, \
         rasterio.open(low_reflectance_image_path) as low_src, \
         rasterio.open(satellite_image_path) as sat_src,\
         rasterio.open(bathy_path) as bathy_src:
        
        # Read the synthesized images and satellite imagery
        high_reflectance_image = high_src.read(masked=True).astype(np.float32)
        low_reflectance_image = low_src.read(masked=True).astype(np.float32)
        satellite_image = sat_src.read(masked=True).astype(np.float32)
        bathymetry_data = bathy_src.read(1, masked=True)

        # When estimating the normalisation scale we need to make an allowance for
        # image noise. Pixels will sometimes be brighter than the estimated high
        # reflectance and some pixels will be darker than the low reflectance due to noise.
        # we therefore slightly expand the range to prevent clipping of these pixels
        # The mapped range is expanded by 2x expand_normalisation_range
        expand_normalisation_range = 0
        
        
        
        # 1.0 - maximum reflectance    example values: 1000+500 = 1500
        # 0.9 - high_reflectance_image                 900+500 = 1400
        # ..
        # 0.3 - low_reflectance_image                  300+500 = 800
        # 0.0 - minimum reflectance                    0 + 500 = 500
        
        # Example range: (1400 - 800)/(0.9-0.3) = 1000
        
        # In shallow areas there is a much greater brightness difference between high
        # and low reflectance patches than in deep areas where the difference between the
        # high reflectance areas and the background rayleigh scattering (brightness of open
        # ocean areas) is very low. We therefore want to scale up the contrast in
        # deep areas. 
        # We estimate the amount of scaling from the difference between the bathymetry
        # based high and low reflectance images. The goal is to scale the images to be between
        # 0 (at or below low reflectance estimate) - 1 (at or above high reflectance estimate).
        reflectance_range = (high_reflectance_image - low_reflectance_image)/(FITTED_HIGH_REFLECTANCE-FITTED_LOW_REFLECTANCE)
        
        # Small errors in the modelling of the background ocean colour might result
        # in deep areas having an inversion of the high and low reflectance image estimates.
        # We therefore want to make sure the range is not zero or negative and so we
        # clip to a minimum value. We are processing in original satellite image integer
        # brightness increments so 1 is a small value.
        
        reflectance_range[reflectance_range < REFLECTANCE_RANGE_LOWER_LIMIT] = REFLECTANCE_RANGE_LOWER_LIMIT
        
        # Calculate what zero reflectance should be.
        # Example range: 800 - 1000 * 0.3 = 500
        zero_reflectance_level = low_reflectance_image - reflectance_range * FITTED_LOW_REFLECTANCE

        
        
        filtered_raster_data = gaussian_filter(satellite_image, sigma=(0, REFLECTANCE_FILTER_SIGMA, REFLECTANCE_FILTER_SIGMA), mode='nearest')
        
        
        # Contrast enhancement that is adjusted by depth.
        # Scale by the range. Values should be nominally between 0 (low reflectance) and
        # 1 (high reflectance), but with lots of noise, so expect values outside this range.
        normalized_reflectance = (filtered_raster_data - zero_reflectance_level) / reflectance_range
        
        
        
        # Clip values to the 0-1 range
        normalized_reflectance = np.clip(normalized_reflectance, 0, 1)
        
        # Scale to 16-bit output, offset by 1 to allow for no_data. Using 8 bit fails because
        # the 4 bands get interpreted as R, G, B, A and the forth band acts as a mask
        # in QGIS.
        normalized_reflectance_16bit = (normalized_reflectance * 254+1).astype(np.uint16)
        
        # Create mask based on depth threshold
        bathymetry_mask = bathymetry_data < bathymetry_threshold
        
        # Make deep areas 0 (no-data value)
        normalized_reflectance_16bit_masked = np.where(bathymetry_mask, 0, normalized_reflectance_16bit)
        
        
        # Add 1 to re
        reflectance_range_uint16 = np.clip(reflectance_range, 0, 65536).astype(np.uint16)
        
        # Save the estimated relative reflectance image for analysis
        if SAVE_REFLECTANCE_RANGE_IMAGE:
            reflectance_range_image_path = REFLECTANCE_RANGE_IMAGE_PATH_TEMPLATE.format(working_path = working_path, swath = swath)
            with rasterio.open(
                reflectance_range_image_path, 'w',
                driver='GTiff',
                height=sat_src.height,
                width=sat_src.width,
                count=sat_src.count,
                dtype=reflectance_range_uint16.dtype,
                crs=sat_src.crs,
                transform=sat_src.transform,
                nodata=0,  # Explicitly set 0 as no-data value
                compress='lzw'
            ) as dst:
                dst.write(reflectance_range_uint16)
                print(f"Saving {reflectance_range_image_path}")
        
        # Save the normalized and scaled reflectance as an 8-bit image
  
        normalized_reflectance_image_path = NORMALIZED_REFLECTANCE_IMAGE_PATH_TEMPLATE.format(output_path = output_path, swath = swath)
        with rasterio.open(
            normalized_reflectance_image_path, 'w',
            driver='GTiff',
            height=sat_src.height,
            width=sat_src.width,
            count=sat_src.count,
            dtype=normalized_reflectance_16bit_masked.dtype,
            crs=sat_src.crs,
            transform=sat_src.transform,
            nodata=0,  # Explicitly set 0 as no-data value
            compress='lzw'
        ) as dst:
            dst.write(normalized_reflectance_16bit_masked)
            print(f"Saving {normalized_reflectance_image_path}")





# Main script
if __name__ == "__main__":

    
    for i in range(len(SENTINEL2_IMG_TILE_FILES)):
        working_path = f'{WORKING_BASE_PATH}/{ANALYSIS_FOLDERS[i]}/'
        output_path = f'{OUTPUT_BASE_PATH}/{ANALYSIS_FOLDERS[i]}/'
        
        print(f'{working_path=}')
        done_file_path = DONE_PATH_TEMPLATE.format(working_path = working_path)
        if os.path.exists(done_file_path):
            print(f'Skipping processing {SENTINEL2_IMG_TILE_FILES[i]} and {done_file_path} exists')
            continue
            
        print(f'----- Processing {SENTINEL2_IMG_TILE_FILES[i]} ------------')
        df = pre_data(bathymetry_file = BATHYMETRY_FILES[i],
            sentinel2_img_tile_file = SENTINEL2_IMG_TILE_FILES[i],
            swath_area_id_list = SWATH_AREA_ID_LISTS[i],
            swath_analysis_areas_file = SWATH_ANALYSIS_AREAS_FILE,
            depth_reflect_sampling_points_file = DEPTH_REFLECTANCE_SAMPLING_POINTS_FILE,
            working_path = working_path,
            output_path = output_path,
            resampling = RESAMPLING,
            temp_path = TEMP_PATH
            )
        print('---- Modelling and synthesising outputs ------')
        model_params_df = fit_models(df, BANDS, output_path, MODEL_PARAM_ADJ_Lbj_K_Lwj_RATIO[i])
        print('Finished fit_models')
        plot_data_and_models(df, BANDS, output_path, model_params_df)
        synthesise_imagery_from_bathy(df, BANDS, working_path, output_path, model_params_df)
        for swath in SWATH_AREA_ID_LISTS[i]:
            process_imagery(swath, model_params_df, working_path, FITTED_HIGH_REFLECTANCE[i], FITTED_LOW_REFLECTANCE[i], BATHY_THRESHOLD)  
        
        # Text to write in the file
        text = """
This file is created at the end of processing this folder. If this file exists, 
then when the analysis is repeated, the processing for this folder is skipped. 
Delete this file or the directory to redo the processing associated with this folder.
"""
        # Writing the text to the file
        with open(done_file_path, 'w') as file:
            file.write(text)