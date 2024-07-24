import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from cdflib.xarray import cdf_to_xarray, xarray_to_cdf

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, met_to_j2000ns, write_cdf

IMAP_EPOCH = np.datetime64("2010-01-01T00:00:00", "ns")

# Create CDF Object and stuff
idex_attrs = ImapCdfAttributes()
idex_attrs.source_dir = Path(__file__).parent.parent / "tests"
idex_attrs.add_instrument_global_attrs("idex")
idex_attrs.add_instrument_variable_attrs("idex", "l1")
logical_source = idex_attrs.get_global_attributes("imap_idex_l1_sci")
attrs = idex_attrs.get_variable_attributes("sample_test", False)
print("ATTRS: ", attrs)

# Create some example data
temperature = np.random.rand(5, 3) * 30  # 5 days, 3 locations
precipitation = np.random.rand(5, 3) * 100  # 5 days, 3 locations

# Create time and location coordinates
times = pd.date_range("2024-07-01", periods=5)
locations = ["Location1", "Location2", "Location3"]

# Convert times to CDF_EPOCH
times_seconds = [(t - IMAP_EPOCH).total_seconds() for t in times]
epoch_time = met_to_j2000ns(times_seconds)  # These are int64 type

print("epoch_time type: ", type(epoch_time[0]))
print("HERE: ", type(epoch_time))

istp_dict = {
    "VAR_TYPE": "data",
    "Project": "Project Input",
    "CATDESC": "Catdesc input",
    "DISPLAY_TYPE": "no_plot",
    "LABLAXIS": "Label",
    "FIELDNAM": "Field",
    "FORMAT": "I10",
    "UNITS": "seconds",
    "VALIDMIN": "-130",
    "VALIDMAX": "130",
    "FILLVAL": "-9223372036854775808",
}

epoch_dict = {
    "VAR_TYPE": "support_data",
    "CATDESC": "Time, number of nanoseconds since J2000 with leap seconds included",
    "FIELDNAM": "epoch",
    "FILLVAL": -9223372036854775808,
    "LABLAXIS": "epoch",
    "FORMAT": "",
    "UNITS": "ns",
    "VALIDMIN": -315575942816000000,
    "VALIDMAX": 3155630469184000000,
    "SCALETYP": "linear",
    "MONOTON": "INCREASE",
    "TIME_BASE": "J2000",
    "TIME_SCALE": "Terrestrial Time",
    "REFERENCE_POSITION": "Rotating Earth Geoid",
    "DISPLAY_TYPE": "no_plot",
}

epoch = xr.DataArray(epoch_time, name="epoch", dims=["epoch"], attrs=epoch_dict)
location = xr.DataArray(locations, name="location", dims=["location"], attrs=istp_dict)

# Create the dataset
data = xr.Dataset(
    {
        "temperature": (["epoch", "location"], temperature, istp_dict),
        "precipitation": (["epoch", "location"], precipitation, istp_dict),
    },
    coords={"epoch": epoch, "location": location},
    attrs=attrs,
)

print(data)
print(data.coords)

# File name
file_name = "example_file.cdf"

# Cleaning away old files
if os.path.exists(file_name):
    os.remove(file_name)

# Using base functions
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
)
cdf_to_xarray(file_name)

file_cdf = write_cdf(data)
load_cdf(file_cdf)
