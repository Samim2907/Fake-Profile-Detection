import pandas as pd

df1 = pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\Datasets\facebook_profiles_7000.csv")
df2 = pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\Datasets\fake_social_media.csv")
df3 = pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\Datasets\Instagram.csv")
df4 = pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\Datasets\twitter_profiles_original.csv")
df5 = pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\Datasets\twitter_profiles.csv")

# ==============================
# CLEAN FACEBOOK DATASET
# ==============================

df1 = df1.rename(columns={
    "followers": "followers_count",
    "following": "following_count",
    "profile_pic": "profile_pic",
    "bio_length": "bio_length",
    "label": "label"
})

df1["post_count"] = 0  
df1 = df1[[
    "followers_count",
    "following_count",
    "post_count",
    "bio_length",
    "profile_pic",
    "label"
]]

# ==============================
# CLEAN FAKE SOCIAL MEDIA DATASET
# ==============================

df2 = df2.rename(columns={
    "followers": "followers_count",
    "following": "following_count",
    "posts": "post_count",
    "has_profile_pic": "profile_pic",
    "bio_length": "bio_length",
    "is_fake": "label"
})

df2 = df2[[
    "followers_count",
    "following_count",
    "post_count",
    "bio_length",
    "profile_pic",
    "label"
]]

# ==============================
# CLEAN INSTAGRAM DATASET
# ==============================

df3 = df3.rename(columns={
    "#followers": "followers_count",
    "#follows": "following_count",
    "#posts": "post_count",
    "profile pic": "profile_pic",
    "description length": "bio_length",
    "fake": "label"
})

df3 = df3[[
    "followers_count",
    "following_count",
    "post_count",
    "bio_length",
    "profile_pic",
    "label"
]]

# ==============================
# CLEAN TWITTER ORIGINAL DATASET
# ==============================

df4 = df4.rename(columns={
    "followers_count": "followers_count",
    "friends_count": "following_count",
    "post_count": "post_count",
    "default_profile_image": "profile_pic",
    "label": "label"
})

df4["bio_length"] = df4["description"].fillna("").apply(len)

df4 = df4[[
    "followers_count",
    "following_count",
    "post_count",
    "bio_length",
    "profile_pic",
    "label"
]]

df4["profile_pic"] = df4["profile_pic"].replace({"True":1,"False":0})

# ==============================
# CLEAN TWITTER SECOND DATASET
# ==============================

df5 = df5.rename(columns={
    "followers_count": "followers_count",
    "friends_count": "following_count",
    "post_count": "post_count",
    "default_profile_image": "profile_pic",
    "label": "label"
})

df5["bio_length"] = df5["description"].fillna("").apply(len)

df5 = df5[[
    "followers_count",
    "following_count",
    "post_count",
    "bio_length",
    "profile_pic",
    "label"
]]

# ==============================
# MERGE DATASETS
# ==============================

combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# remove duplicates
combined_df.drop_duplicates(inplace=True)

# ==============================
# CREATE EXTRA FEATURES
# ==============================

combined_df["follower_following_ratio"] = combined_df["followers_count"] / (combined_df["following_count"] + 1)

# ==============================
# SAVE CLEAN DATASET
# ==============================

combined_df.to_csv("cleaned_fake_profiles_dataset.csv", index=False)

print("Dataset cleaned and saved successfully!")
print(combined_df.head())
print("\nDataset shape:", combined_df.shape)