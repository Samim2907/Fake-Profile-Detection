# 🔍 Fake Profile Detection

A machine learning system that detects fake social media profiles using a Stacking Ensemble classifier trained on ~29,700 profiles from Twitter, Instagram, and Facebook.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy | 97.35% |
| Precision (Real) | 0.97 |
| Precision (Fake) | 0.98 |
| Recall (Real) | 0.99 |
| Recall (Fake) | 0.95 |
| Decision Threshold | 0.70 |

---

## 🧠 Model Architecture

**Stacking Ensemble** with:
- Random Forest (n=300) — base learner
- Extra Trees (n=200) — base learner
- Gradient Boosting (n=100, depth=5) — base learner
- Logistic Regression — meta learner (cv=5)

---

## 📁 Dataset Sources

| Dataset | Profiles | Platform |
|---|---|---|
| twitter_profiles.csv | 10,000 | Twitter |
| twitter_profiles_original.csv | 7,698 | Twitter |
| Instagram.csv | 5,000 | Instagram |
| facebook_profiles_7000.csv | 7,000 | Facebook |
| **Total** | **~29,700** | |

> `fake_social_media.csv` was excluded due to severe class imbalance (2,993 fake vs 7 real).

---

## 🔧 Features Used (19 total)

**Raw features:**
- `followers_count`, `following_count`, `post_count`
- `bio_length`, `default_profile_image`, `verified`
- `account_age_days`, `follower_following_ratio`, `username_num_ratio`

**Engineered features:**
- `posts_per_follower`, `posts_per_following`, `engagement_proxy`
- `low_followers_flag`, `high_following_flag`, `suspicious_ratio_flag`
- `has_bio`, `no_posts_flag`, `ff_ratio_clipped`, `has_bg_image`

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Samim2907/Fake-Profile-Detection.git
cd Fake-Profile-Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train and save the model
Place the 4 CSV files in the project folder, then run:
```bash
python train_and_save.py
```
This generates `model.pkl`, `scaler.pkl`, and `features.pkl`.

### 4. Launch the app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## 🖥️ App Features

- **Manual Input tab** — enter profile details and get an instant prediction with confidence %
- **Batch CSV Upload tab** — upload multiple profiles and download results
- **Confidence gauge** — visual Plotly gauge showing fake probability
- **Sidebar** — model info, accuracy, and full feature list

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
plotly
```

---

## ⚠️ Limitations

- Only 19 statistical features — no content, text, or network analysis
- Does not support live API scraping from Instagram, Facebook, or X
- May produce false positives on fan accounts with moderate follower counts
- Model performance reflects the distribution of the training datasets

---

## 🔮 Future Work

- Integrate X (Twitter) API for live profile fetching
- Add NLP features from bio and post content (BERT embeddings)
- Graph-based features from follower/following network topology
- Continuous retraining pipeline as fake account patterns evolve

---

## 👤 Author

**Samim** — [@Samim2907](https://github.com/Samim2907)
