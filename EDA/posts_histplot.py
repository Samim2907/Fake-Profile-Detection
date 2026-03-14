import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df =  pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\cleaned_fake_profiles_dataset.csv")

plt.figure()
sns.histplot(np.log1p(df["post_count"]), bins=50)
plt.xlabel("log(Posts + 1)")
plt.title("Post Count Distribution (Log Scale)")
plt.show()