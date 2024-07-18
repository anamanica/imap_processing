import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.utils import load_cdf, write_cdf

# Create some example data
temperature = np.random.rand(5, 3) * 30  # 5 days, 3 locations
precipitation = np.random.rand(5, 3) * 100  # 5 days, 3 locations

# Create time and location coordinates
times = pd.date_range("2024-07-01", periods=5)
locations = ["Location1", "Location2", "Location3"]

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
    attrs={
        "FILLVAL": "hello",
        "Logical_source": "imap_codice_l1a_hskp",
        "value_3": "value_3_input",
    },
)


def testing_data_to_cdf():
    file_name = write_cdf(data)
    l1_data = load_cdf(file_name)
    assert (l1_data["TOF_High"][13].data == data).all()
