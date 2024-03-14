"""
Australian Institute of Marine Science
Eric Lawrey

This script downloads the datasets needed in this analysis. Note the imagery from
Google Earth Engine is not available
"""
from data_downloader import DataDownloader
import os

data_cache_path = "working/data-cache"
# Create an instance of the DataDownloader class
downloader = DataDownloader(download_path=data_cache_path)


print("Downloading source data files. This will take a while ...")


# -------------------------------------------------------
# GBR Bathymetry 30 m 2020
# Citation: Beaman, R.J. 2017. High-resolution depth model for the Great Barrier Reef - 30 m. 
#   Geoscience Australia, Canberra. http://dx.doi.org/10.4225/25/5a207b36022d2
# Licence: Creative Commons Attribution 4.0 International Licence http://creativecommons.org/licenses/
# Metadata: http://pid.geoscience.gov.au/dataset/ga/115066
# Direct download: https://files.ausseabed.gov.au/survey/Great%20Barrier%20Reef%20Bathymetry%202020%2030m.zip

downloader.download_and_unzip(
    'https://files.ausseabed.gov.au/survey/Great%20Barrier%20Reef%20Bathymetry%202020%2030m.zip', 
    'GBR_GA_Great-Barrier-Reef-Bathy-30m_2020'
)

# -------------------------------------------------------
# GBR Bathymetry 100 m 2020
# Citation: Beaman, R.J. 2020. High-resolution depth model for the Great Barrier Reef and Coral Sea - 100 m. 
#   Geoscience Australia, Canberra. http://dx.doi.org/10.26186/5e2f8bb629d07
# Licence: Creative Commons Attribution 4.0 International Licence http://creativecommons.org/licenses/
# Metadata: http://pid.geoscience.gov.au/dataset/ga/133163
# Direct download: https://files.ausseabed.gov.au/survey/Great%20Barrier%20Reef%20Bathymetry%202020%20100m.zip

downloader.download_and_unzip(
    'https://files.ausseabed.gov.au/survey/Great%20Barrier%20Reef%20Bathymetry%202020%20100m.zip', 
    'GBR_GA_Great-Barrier-Reef-Bathy-100m_2020'
)


# Download the true colour image of North Flinders Reef. This is our reference
# reef and we want to demonstrate what it looks like in TrueColour.
downloader.download(f'https://nextcloud.eatlas.org.au/s/NjbyWRxPoBDDzWg/download?path=%2Flossless%2FCoral-Sea%2FS2_R1_TrueColour&files=CS_AIMS_Coral-Sea-Features_Img_S2_R1_TrueColour_55KFA.tif', f'{data_cache_path}/S2_R1_TrueColour/CS_AIMS_Coral-Sea-Features_Img_S2_R1_TrueColour_55KFA.tif')


#----------------------------------------------------
# Lawrey, E., Hammerton, M. (2024). Marine satellite imagery test collections (AIMS) [Data set]. eAtlas.
# https://doi.org/10.26274/zq26-a956
direct_download_url = 'https://nextcloud.eatlas.org.au/s/9tbZP8Rbk5FxiQ6/download?path=%2FCS_NESP-MaC-2-3_AIMS_Benth-reflect'
# Define the patterns to search for
patterns = [
    'CS_NESP-MaC-2-3_AIMS_Benth-reflect/*',
]
# Use this approach as the zip file contains an internal CS_NESP-MaC-2-3_AIMS_Benthic-reflectance/ that makes
# the overall paths too long. 
downloader.download_unzip_keep_subset(direct_download_url, patterns, 'S2_Benth-Ref')
