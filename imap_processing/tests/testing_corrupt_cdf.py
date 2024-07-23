import os

import numpy as np
import pandas as pd
import xarray as xr
from cdflib.xarray import xarray_to_cdf

# COMPLEX EXAMPLE USAGE
# # Create CDF Object and stuff
# idex_attrs = ImapCdfAttributes()
# idex_attrs.add_instrument_global_attrs("idex")
# idex_attrs.add_instrument_variable_attrs("idex", "l1")
# attrs = idex_attrs.get_global_attributes(
#     "imap_idex_l1_sci"
# ) | idex_attrs.get_variable_attributes("sample_rate_base")
# print(attrs)

# Create some example data
temperature = np.random.rand(5, 3) * 30  # 5 days, 3 locations
precipitation = np.random.rand(5, 3) * 100  # 5 days, 3 locations

# Create time and location coordinates
times = pd.date_range("2024-07-01", periods=5)
locations = ["Location1", "Location2", "Location3"]

# Creating a stupid string
# with open("beeMovie.txt") as file:
#     beeMovie = file.read().replace("\n", " ")

# Create the dataset
data = xr.Dataset(
    {
        "temperature": (["time", "location"], temperature),
        "precipitation": (["time", "location"], precipitation),
    },
    coords={
        "time": times,
        "location": locations,
    },
    # attrs=attrs, COMPLEX
    attrs={"FILLVAL": "-1.0e31", "VALIDMAX": 130},
)

# Ensure all attribute values are strings to avoid issues
for attr in data.attrs:
    data.attrs[attr] = str(data.attrs[attr])

print(data)

# File name
file_name = "example_file.cdf"

# Cleaning away old files
if os.path.exists(file_name):
    os.remove(file_name)

xarray_to_cdf(
    data,
    file_name,
    unix_time_to_cdf_time=False,
    istp=False,
    terminate_on_warning=False,
    auto_fix_depends=True,
    record_dimensions=["record0"],
    compression=0,
    nan_to_fillval=True,
    from_unixtime=False,
    from_datetime=False,
    unixtime_to_cdftt2000=False,
    datetime_to_cdftt2000=True,
    datetime64_to_cdftt2000=True,
)

# cdf_to_xarray(file_name)
print("Passed gen functions")

# COMPLEX USAGE
# cdf_file = write_cdf(data)
# load_cdf(cdf_file)
