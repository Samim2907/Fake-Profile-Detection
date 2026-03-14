import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df =  pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\cleaned_fake_profiles_dataset.csv")

sns.scatterplot(
    x=np.log1p(df["following_count"]),
    y=np.log1p(df["followers_count"]),
    hue=df["label"],
    alpha=0.5
)

plt.xlabel("log(Following + 1)")
plt.ylabel("log(Followers + 1)")
plt.title("Followers vs Following (Log Scale)")
plt.show()