# INTRODUCTION




# SETUP


import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt


# LOAD THE DATA


master_url_root_type_str = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

df_small_noise_url_suffix_type_str = "artificialNoAnomaly/art_daily_small_noise.csv"

df_small_noise_url_type_str = master_url_root_type_str + df_small_noise_url_suffix_type_str

df_small_noise_type_DataFrame = pd.read_csv(
    df_small_noise_url_type_str, parse_dates=True, index_col="timestamp"
)

df_daily_jumpsup_url_suffix_type_str = "artificialWithAnomaly/art_daily_jumpsup.csv"

df_daily_jumpsup_url_type_str = master_url_root_type_str + df_daily_jumpsup_url_suffix_type_str

df_daily_jumpsup_type_DataFrame = pd.read_csv(
    df_daily_jumpsup_url_type_str, parse_dates=True, index_col="timestamp"
)


# QUICK LOOK AT THE DATA


print(df_small_noise_type_DataFrame.head())

# timestamp ie date and I think minute with a value next to it

print(df_daily_jumpsup_type_DataFrame.head())


# VISUALISE THE DATA


# TIMESERIES DATA WITHOUT ANOMALIES


fig_type_Figure, ax_type_AxesSubplot = plt.subplots()
df_small_noise_type_DataFrame.plot(legend=False, ax=ax_type_AxesSubplot)
plt.show()


# TIMESERIES DATA WITH ANOMALIES


fig_type_Figure, ax_type_AxesSubplot = plt.subplots()
df_daily_jumpsup_type_DataFrame.plot(legend=False, ax=ax_type_AxesSubplot)
plt.show()


# PREPARE TRAINING DATA


# Normalize and save the mean and std we get,
# for normalizing test data.
training_mean_type_Series = df_small_noise_type_DataFrame.mean()
training_std_type_Series = df_small_noise_type_DataFrame.std()
df_training_value_type_DataFrame = (df_small_noise_type_DataFrame - training_mean_type_Series) / training_std_type_Series
print("Number of training samples:", len(df_training_value_type_DataFrame))


# CREATE SEQUENCES


