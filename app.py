import re
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import nltk
from nltk.corpus import stopwords

# -----------------------
# App setup
# -----------------------

st.set_page_config(page_title="Steam Review Analyzer", layout="wide")
st.title("Steam Review Analyzer")
st.write(
    "This app fetches **Steam reviews** for a selected game and analyzes what textual characteristics are associated with positive and negative ratings."
)

# -----------------------
# NLTK stopwords (English)
# -----------------------

@st.cache_resource
def load_stopwords():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))

STOPWORDS = load_stopwords()

# -----------------------
# Helper functions
# -----------------------

def extract_appid(text: str):
    """
    Accepts either a numeric AppID or a Steam store URL.
    """
    text = text.strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"/app/(\d+)", text)
    if match:
        return int(match.group(1))
    return None

@st.cache_data(show_spinner=False)
def fetch_steam_reviews(appid: int, max_reviews: int = 300, sleep_s: float = 0.2):
    """
    Fetch reviews using Steam's public reviews endpoint.
    Returns a DataFrame with review text and labels.
    """
    url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {
        "json": 1,
        "language": "english",
        "filter": "recent",
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": 100,
        "cursor": "*"
    }

    rows = []
    fetched = 0

    while fetched < max_reviews:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        reviews = data.get("reviews", [])
        if not reviews:
            break

        for rev in reviews:
            rows.append({
                "review": rev.get("review", ""),
                "label": int(rev.get("voted_up", False)),  # 1 = positive, 0 = negative
            })
            fetched += 1
            if fetched >= max_reviews:
                break

        params["cursor"] = data.get("cursor", params["cursor"])
        time.sleep(sleep_s)

    return pd.DataFrame(rows)

def tokenize(text: str):
    return re.findall(r"[A-Za-z]+", text.lower())

def top_words_by_class(df, top_n=20):
    pos_tokens = []
    neg_tokens = []

    for _, row in df.iterrows():
        tokens = [
            t for t in tokenize(row["review"])
            if t not in STOPWORDS and len(t) > 2
        ]
        if row["label"] == 1:
            pos_tokens.extend(tokens)
        else:
            neg_tokens.extend(tokens)

    pos_counts = pd.Series(pos_tokens).value_counts().head(top_n)
    neg_counts = pd.Series(neg_tokens).value_counts().head(top_n)

    return pos_counts, neg_counts

# -----------------------
# UI inputs
# -----------------------

appid_input = st.text_input(
    "Steam AppID or Store Link",
    value="620"  # Portal 2 as an example
)

appid = extract_appid(appid_input)

col1, col2 = st.columns(2)
with col1:
    max_reviews = st.number_input(
        "Max reviews to fetch",
        min_value=50,
        max_value=2000,
        value=300,
        step=50
    )
with col2:
    sleep_s = st.slider(
        "Politeness delay (seconds)",
        0.0, 1.0, 0.2, 0.05
    )

# -----------------------
# Main action
# -----------------------

if st.button("Fetch and analyze reviews"):
    if appid is None:
        st.error("Could not detect AppID. Enter a number (e.g. 620) or paste a Steam store link.")
        st.stop()

    with st.spinner("Fetching reviews..."):
        df = fetch_steam_reviews(appid, int(max_reviews), float(sleep_s))

    if df.empty:
        st.warning("No reviews were returned.")
        st.stop()

    # Basic text features

    df["word_len"] = df["review"].astype(str).apply(lambda x: len(tokenize(x)))
    df["char_len"] = df["review"].astype(str).str.len()

    st.success(f"Fetched {len(df)} reviews.")
    st.subheader("Preview")
    st.dataframe(df.head(15), use_container_width=True)

    # -----------------------
    # A) Descriptive analytics
    # -----------------------

    st.subheader("A) Descriptive analytics")

    pos_rate = df["label"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total reviews", len(df))
    c2.metric("Positive reviews", f"{pos_rate:.2%}")
    c3.metric("Negative reviews", f"{1 - pos_rate:.2%}")

    length_stats = df.groupby("label")[["word_len", "char_len"]].mean()
    length_stats.index = ["Negative", "Positive"]
    st.write("Average review length by class:")
    st.dataframe(length_stats, use_container_width=True)

    fig = plt.figure()
    plt.hist(df[df["label"] == 1]["word_len"], bins=20, alpha=0.7, label="Positive")
    plt.hist(df[df["label"] == 0]["word_len"], bins=20, alpha=0.7, label="Negative")
    plt.xlabel("Words per review")
    plt.ylabel("Count")
    plt.legend()
    st.pyplot(fig)

    # -----------------------
    # B) Top words
    # -----------------------

    st.subheader("B) Top words by rating")

    top_pos, top_neg = top_words_by_class(df)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Most common words in **positive** reviews")
        st.dataframe(top_pos.rename("count"), use_container_width=True)
    with c2:
        st.write("Most common words in **negative** reviews")
        st.dataframe(top_neg.rename("count"), use_container_width=True)

    # -----------------------
    # C) Simple classifier
    # -----------------------

    st.subheader("C) Text classification (TF-IDF + Logistic Regression)")

    X = df["review"].astype(str).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOPWORDS),
        max_features=5000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1", f"{f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion matrix (rows = true, columns = predicted):")
    st.write(pd.DataFrame(cm, index=["True Neg", "True Pos"], columns=["Pred Neg", "Pred Pos"]))

    # Top coefficients
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_pos_idx = np.argsort(coefs)[-15:][::-1]
    top_neg_idx = np.argsort(coefs)[:15]

    c1, c2 = st.columns(2)
    with c1:
        st.write("Words most associated with **positive** reviews")
        st.dataframe(
            pd.DataFrame({"word": feature_names[top_pos_idx], "coef": coefs[top_pos_idx]}),
            use_container_width=True
        )
    with c2:
        st.write("Words most associated with **negative** reviews")
        st.dataframe(
            pd.DataFrame({"word": feature_names[top_neg_idx], "coef": coefs[top_neg_idx]}),
            use_container_width=True
        )

    st.caption("Steam Review Analyzer")
