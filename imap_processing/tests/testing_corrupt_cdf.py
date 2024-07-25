import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cdflib.xarray import cdf_to_xarray, xarray_to_cdf

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, met_to_j2000ns, write_cdf

# Constant to create the correct type of epoch
IMAP_EPOCH = np.datetime64("2010-01-01T00:00:00", "ns")


@pytest.fixture()
def data():
    # Create CDF Object for cdf attributes
    idex_attrs = ImapCdfAttributes()
    idex_attrs.source_dir = Path(__file__).parent.parent / "tests"
    idex_attrs.add_instrument_global_attrs("idex")
    idex_attrs.add_instrument_variable_attrs("idex", "l1")

    # Create some example data
    temperature = np.random.rand(5, 3) * 30  # 5 days, 3 locations

    # Create time and location coordinates
    times = pd.date_range("2024-07-01", periods=5)
    locations = ["Location1", "Location2", "Location3"]

    # Convert times to CDF_EPOCH
    times_seconds = [(t - IMAP_EPOCH).total_seconds() for t in times]
    epoch_time = met_to_j2000ns(times_seconds)  # These are int64 type

    epoch = xr.DataArray(
        epoch_time,
        name="epoch",
        dims=["epoch"],
        attrs=idex_attrs.get_variable_attributes("epoch", False),
    )
    location = xr.DataArray(
        locations,
        name="location",
        dims=["location"],
        attrs=idex_attrs.get_variable_attributes("location_attrs", False),
    )

    # Create the dataset
    data = xr.Dataset(
        {
            "temperature": (
                ["epoch", "location"],
                temperature,
                idex_attrs.get_variable_attributes("temperature", False),
            ),
        },
        coords={"epoch": epoch, "location": location},
        attrs=idex_attrs.get_variable_attributes("global_stuff", False),
    )
    return data


def test_cdflib_calls(data):
    # Dealing with files for cdflib calls
    file_name = "example_file.cdf"
    # Cleaning away old files
    if os.path.exists(file_name):
        os.remove(file_name)

    # cdflib calls
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


@pytest.mark.xfail()  # Comment this out to see full error!
def testing_utils_calls(data):
    # Dealing with files for utils calls
    data_version = data.attrs["Data_version"]
    file_name_2 = (
        "/Users/anma6676/Desktop/Repositories/imap_processing/imap_processing/tests/data/imap/idex/l1/2024/07/"
        "imap_idex_l1_sci_20240701_" + data_version + ".cdf"
    )
    if os.path.exists(file_name_2):
        os.remove(file_name_2)

    # Utils calls
    file_cdf = write_cdf(data)
    load_cdf(file_cdf)


@pytest.fixture()
def written_data():
    global_stuff = {
        "Logical_source": "imap_idex_l1_sci",
        "Data_version": "v123",
        "Project": "Project Input",
        "Source_name": "Source input",
        "Discipline": "Discipline input",
        "Data_type": "no_plot",
        "Descriptor": "Descriptor input",
        "PI_name": "Ana Manica",
        "PI_affiliation": "Affiliation of cool peeps",
        "TEXT": "Text input",
        "Instrument_type": "Harp",
        "Mission_group": "Cats",
        "Logical_source_description": "no",
    }

    epoch_dict = {
        "CATDESC": "Time, number of nanoseconds since J2000 with leap seconds included",
        "FIELDNAM": "epoch",
        "FILLVAL": -9223372036854775808,
        "FORMAT": "",
        "UNITS": "ns",
        "VALIDMIN": -315575942816000000,
        "VALIDMAX": 3155630469184000000,
        "VAR_TYPE": "support_data",
    }

    location_attrs = {
        "VAR_TYPE": "support_data",
        "CATDESC": "",
        "FIELDNAM": "",
        "FORMAT": "",
        "UNITS": "",
    }

    temperature_dict = {
        "VAR_TYPE": "data",
        "CATDESC": "Catdesc input",
        "DISPLAY_TYPE": "no_plot",
        "LABLAXIS": "Label",
        "FIELDNAM": "Field",
        "FORMAT": "F64.5",
        "UNITS": "seconds",
        "VALIDMIN": -130,
        "VALIDMAX": 130,
        "FILLVAL": -1.0e31,
        "DEPEND_0": "epoch",
        "DEPEND_1": "location",
    }
    # Create some example data
    temperature = np.random.rand(5, 3) * 30  # 5 days, 3 locations

    # Create time and location coordinates
    times = pd.date_range("2024-07-01", periods=5)
    locations = ["Location1", "Location2", "Location3"]

    # Convert times to CDF_EPOCH
    times_seconds = [(t - IMAP_EPOCH).total_seconds() for t in times]
    epoch_time = met_to_j2000ns(times_seconds)  # These are int64 type

    epoch = xr.DataArray(
        epoch_time,
        name="epoch",
        dims=["epoch"],
        attrs=epoch_dict,
    )
    location = xr.DataArray(
        locations,
        name="location",
        dims=["location"],
        attrs=location_attrs,
    )

    # Create the dataset
    written_data = xr.Dataset(
        {
            "temperature": (
                ["epoch", "location"],
                temperature,
                temperature_dict,
            ),
        },
        coords={"epoch": epoch, "location": location},
        attrs=global_stuff,
    )

    return written_data


def test_cdflib_calls_written(written_data):
    # Dealing with files for cdflib calls
    file_name = "example_file.cdf"
    # Cleaning away old files
    if os.path.exists(file_name):
        os.remove(file_name)

    # cdflib calls
    xarray_to_cdf(
        written_data,
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


# Now this one. This one DOESN'T fail. Isn't that weird?
def testing_utils_calls_written(written_data):
    # Dealing with files for utils calls
    file_name_2 = (
        "/Users/anma6676/Desktop/Repositories/imap_processing/imap_processing/tests/data/imap/idex/l1/2024/07/"
        "imap_idex_l1_sci_20240701_v123.cdf"
    )
    if os.path.exists(file_name_2):
        os.remove(file_name_2)

    # Utils calls
    file_cdf = write_cdf(written_data)
    load_cdf(file_cdf)
