import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from cdflib.xarray import cdf_to_xarray, xarray_to_cdf

# Create a time coordinate
times = pd.date_range("2024-07-01", periods=5, freq="D")

# Example data arrays
data_var = np.random.rand(5)

# Create the dataset
data = xr.Dataset(
    {"data_var": (["epoch"], data_var)},
    coords={"epoch": ("epoch", times)},
    attrs={
        "Project": "STP>Solar-Terrestrial Physics",
        "Source_name": "IMAP>Interstellar Mapping and Acceleration Probe",
        "Discipline": "Solar Physics>Heliospheric Physics",
        "Mission_group": "IMAP>Interstellar Mapping and Acceleration Probe",
        "PI_name": "Dr. David J. McComas",
        "PI_affiliation": "Princeton Plasma Physics Laboratory, "
        "100 Stellarator Road, Princeton, NJ 08540",
        "File_naming_convention": "source_descriptor_datatype_yyyyMMdd_vNNN",
        "Data_level": "L0",
        "Data_type": "L1A_hist > Level - 1A histogram",
        "Logical_source": "imap_glows_l1a_hist",
        "Logical_source_description": "IMAP Mission GLOWS Histogram Level - 1A Data.",
        "Descriptor": "GLOWS>GLObal Solar Wind Structure",
        "Instrument_type": "Particles (space)",
        "Data_version": "v001",
        "Logical_file_id": "input",
        "TEXT": "Here is some text",
    },
)

# Add ISTP compliant attributes
data["epoch"].attrs["CATDESC"] = "Time"
data["epoch"].attrs["UNITS"] = "ns"
data["epoch"].attrs["FILLVAL"] = -9223372036854775808  # TT2000 fill value
data["epoch"].attrs["VAR_TYPE"] = "data"
data["epoch"].attrs["DISPLAY_TYPE"] = "no_plot"
data["epoch"].attrs["FIELDNAM"] = "FieldName"
data["epoch"].attrs["FORMAT"] = "I10"
data["epoch"].attrs["LABLAXIS"] = "Time"
data["epoch"].attrs["VALIDMIN"] = -315575942816000000
data["epoch"].attrs["VALIDMAX"] = 3155630469184000000
data["data_var"].attrs["CATDESC"] = "Example Data"
data["data_var"].attrs["UNITS"] = "units"
data["data_var"].attrs["FILLVAL"] = -1.0e31
data["data_var"].attrs["VALIDMIN"] = -315575942816000000
data["data_var"].attrs["VALIDMAX"] = 315575942816000000
data["data_var"].attrs["FORMAT"] = "A24"
data["data_var"].attrs["LABLAXIS"] = "Time"
data["data_var"].attrs["DEPEND_0"] = "epoch"
data["data_var"].attrs["VAR_TYPE"] = "support_data"
data["data_var"].attrs["FIELDNAM"] = "FieldName"


# Write to CDF
def write_cdf(dataset: xr.Dataset, file_path: str) -> Path:
    if os.path.exists(file_path):
        os.remove(file_path)
    xarray_to_cdf(dataset, file_path, terminate_on_warning=True)
    return Path(file_path)


# Function to load a CDF file into an xarray Dataset
def load_cdf(file_path: Path, remove_xarray_attrs: bool = True, **kwargs) -> xr.Dataset:
    dataset = cdf_to_xarray(file_path, **kwargs)
    if remove_xarray_attrs:
        for var in dataset.variables:
            for attr in ["plot", "plot_hint"]:
                dataset[var].attrs.pop(attr, None)
    return dataset


def testing_data_to_cdf():
    file_path = "example_istp_compliant.cdf"
    write_cdf(data, file_path)
    l1_data = load_cdf(file_path)
    assert (l1_data["epoch"].data == data["epoch"].data).all()


# def testing_data_to_cdf():
#     file_path = "test.cdf"
#     file_name = write_cdf(data, file_path)
#     l1_data = load_cdf(file_path)
#     assert (l1_data["temperature"].data == data["temperature"].data).all()
#     assert (l1_data["precipitation"].data == data["precipitation"].data).all()
