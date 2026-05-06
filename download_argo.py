#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_argo.py

Argo data download helper script (minimal).

Notes:
- Do NOT hardcode any credentials (usernames/passwords/tokens) in this repository.
- Edit the spatial/temporal bounds and variables as needed.
- This script downloads NetCDF files as optional raw inputs for downstream preprocessing.

Requirements:
  pip install copernicusmarine

Example:
  python download_argo.py

Outputs (default):
  - physical_data_2014to2024.nc
  - biogeochemical_data_2014to2024.nc
"""

import copernicusmarine

# Spatial/temporal bounds (edit as needed)
start_date = "2014-01-01"
end_date = "2024-12-31"
min_lon = -120.0
max_lon = -110.0
min_lat = 20.0
max_lat = 40.0

print("[INFO] Downloading physical data...")
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["thetao", "so", "uo", "vo", "wo", "zos"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=min_lon,
    maximum_longitude=max_lon,
    minimum_latitude=min_lat,
    maximum_latitude=max_lat,
    minimum_depth=0,
    maximum_depth=1000,
    output_filename="physical_data_2014to2024.nc",
)

print("[INFO] Downloading biogeochemical data...")
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_bgc_my_0.25_P1D-m",
    variables=["no3", "po4", "si", "o2", "chl"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=min_lon,
    maximum_longitude=max_lon,
    minimum_latitude=min_lat,
    maximum_latitude=max_lat,
    minimum_depth=0,
    maximum_depth=1000,
    output_filename="biogeochemical_data_2014to2024.nc",
)

print("[OK] Download complete.")