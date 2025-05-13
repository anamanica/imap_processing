"""Code to create quicklook plots."""

import os

import matplotlib.pyplot as plt

import imap_processing


def general_quicklook(variables: list[str], file_name: str) -> None:
    """
    General plotting function. Handles one, or multiple variables.

    Parameters
    ----------
    variables : list[str]
        Desired variable(s) to plot.
    file_name : str
        Desired file to pull variable(s) from.

    Returns
    -------
    None
        This function returns nothing.

    Notes
    -----
    One plotting function to rule them all
    """
    _, instrument, level, _, year_month, version_no = file_name.split("_")
    year = year_month[:4]
    month = year_month[4:]

    # Define file_path
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "imap",
        instrument,
        level,
        year,
        month,
    )
    full_path = os.path.join(file_path, file_name)

    # Check if file exists
    if not os.path.exists(full_path):
        print("File does not exist")
    else:
        # Create and plot dataset
        data_set = imap_processing.cdf.utils.load_cdf(full_path)

        # Ensure that variable is present in dataset
        existing_vars = [var for var in variables if var in data_set]
        missing_vars = [var for var in variables if var not in data_set]

        if missing_vars:
            print("The following were not found in given file: ", missing_vars)

        if not existing_vars:
            print("No given variables found in given file.")
            return

        fig, axes = plt.subplots(
            nrows=len(existing_vars), figsize=(10, 3 * len(existing_vars))
        )
        if len(existing_vars) == 1:
            axes = [axes]

        for index, var in enumerate(existing_vars):
            data_set[var].plot(ax=axes[index])

        plt.tight_layout()
        plt.show()
