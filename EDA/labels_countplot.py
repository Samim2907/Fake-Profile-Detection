import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df =  pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\cleaned_fake_profiles_dataset.csv")

sns.countplot(x="label", data=df)
plt.xticks([0,1], ["Legitimate","Fake"])
plt.title("Class Distribution")
plt.show()