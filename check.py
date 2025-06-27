

import numpy as np
import pandas as pd


def are_csvs_identical(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    return df1.equals(df2)

# Example usage:
file1 = "results/ALL_capped.csv"
file2 = "results/ALL_no_cap.csv"
print("Files 1 2 are identical:", are_csvs_identical(file1, file2))


# Example usage:
file1 = "results/PAK_capped.csv"
file2 = "results/PAK_no_cap.csv"
print("Files are identical:", are_csvs_identical(file1, file2))