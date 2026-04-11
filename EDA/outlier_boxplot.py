import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df =  pd.read_csv(rf"E:\Projects\ML\Fake Profile Detection\cleaned_fake_profiles_dataset.csv")

plt.figure()
sns.boxplot(x="label", y="followers_count", data=df)
plt.title("Followers by Account Type")
plt.show()