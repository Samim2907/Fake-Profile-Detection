from IPython.display import display
import seaborn as sns
import pandas as pd
df =  pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\cleaned_fake_profiles_dataset.csv")
display(df.head())
print(df.shape)