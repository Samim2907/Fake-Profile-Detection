import pandas as pd
import numpy as np
# Load raw datasets ──────────────────────────────────────────────────────
BASE = r"E:\Projects\ML\Fake Profile Detection"

tw      = pd.read_csv(rf"E:\Projects\ML\Fake Profile Detection\Datasets\twitter_profiles.csv")
tw_orig = pd.read_csv(rf"E:\Projects\ML\Fake Profile Detection\Datasets\twitter_profiles_original.csv")
insta   = pd.read_csv(rf"E:\Projects\ML\Fake Profile Detection\Datasets\Instagram.csv")
fb      = pd.read_csv(rf"E:\Projects\ML\Fake Profile Detection\Datasets\facebook_profiles_7000.csv")

# Standardise each dataset ───────────────────────────────────────────────
def build_twitter(df):
    df = df.copy()
    df["created_at"]     = pd.to_datetime(df["created_at"], errors="coerce")
    df["account_age_days"] = (pd.Timestamp("2024-01-01") - df["created_at"]).dt.days
    df["bio_length"]     = df["description"].fillna("").apply(len)
    df["follower_following_ratio"] = df["followers_count"] / (df["friends_count"] + 1)
    for col in ["default_profile_image", "profile_use_background_image", "verified"]:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int)
    out = df[["followers_count", "friends_count", "post_count", "bio_length",
              "default_profile_image", "profile_use_background_image", "verified",
              "account_age_days", "follower_following_ratio", "label"]].copy()
    out.columns = ["followers_count", "following_count", "post_count", "bio_length",
                   "default_profile_image", "has_bg_image", "verified",
                   "account_age_days", "follower_following_ratio", "label"]
    out["username_num_ratio"] = np.nan
    return out

tw_clean      = build_twitter(tw)
tw_orig_clean = build_twitter(tw_orig)

insta_clean = insta.rename(columns={
    "#followers": "followers_count", "#follows": "following_count", "#posts": "post_count",
    "description length": "bio_length", "profile pic": "profile_pic",
    "nums/length username": "username_num_ratio", "fake": "label",
    "followers_following_ratio": "follower_following_ratio",
}).copy()
insta_clean["default_profile_image"] = 1 - insta_clean["profile_pic"]
insta_clean["has_bg_image"] = 0
insta_clean["verified"]     = 0
insta_clean["account_age_days"] = np.nan
insta_clean = insta_clean[["followers_count", "following_count", "post_count", "bio_length",
                            "default_profile_image", "has_bg_image", "verified",
                            "account_age_days", "follower_following_ratio", "label", "username_num_ratio"]]

fb_clean = fb.rename(columns={"followers": "followers_count", "following": "following_count"}).copy()
fb_clean["post_count"]            = 0
fb_clean["default_profile_image"] = 1 - fb_clean["profile_pic"]
fb_clean["has_bg_image"]          = 0
fb_clean["verified"]              = 0
fb_clean["account_age_days"]      = np.nan
fb_clean["follower_following_ratio"] = fb_clean["followers_count"] / (fb_clean["following_count"] + 1)
fb_clean["username_num_ratio"]    = np.nan
fb_clean = fb_clean[["followers_count", "following_count", "post_count", "bio_length",
                      "default_profile_image", "has_bg_image", "verified",
                      "account_age_days", "follower_following_ratio", "label", "username_num_ratio"]]

# ==============================
# MERGE DATASETS
# ==============================

combined_df = pd.concat([tw_clean,tw_orig_clean,insta_clean,fb_clean], ignore_index=True)

combined_df.drop_duplicates(inplace=True)

combined_df.to_csv("cleaned_fake_profiles_dataset.csv", index=False)

print("Dataset cleaned and saved successfully!")
print(combined_df.head())
print("\nDataset shape:", combined_df.shape)