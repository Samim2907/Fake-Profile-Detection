import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df =  pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\cleaned_fake_profiles_dataset.csv")

df = df.dropna()
print(df.isnull().sum())
df = df.drop_duplicates()
print(df.shape)
df.to_csv(r"E:\Projects\ML\Fake Profile Detection\complete_dataset.csv", index=False)