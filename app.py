import streamlit as st
import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')


# Load Dataset

df = pd.read_csv("reviews.csv")

# Preprocessing

def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df['clean_review'] = df['review'].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_review'])
y = df['label']

# Train Model
model = LogisticRegression()
model.fit(X, y)


# UI Design

st.title("🛒 Fake Review Detection System")

st.write("Enter a product review to check whether it is Fake or Genuine.")

user_input = st.text_area("✍️ Enter Review")

if st.button("Check Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        clean = preprocess(user_input)
        vector = vectorizer.transform([clean])
        result = model.predict(vector)

        if result[0] == 1:
            st.success("✅ Genuine Review")
        else:
            st.error("❌ Fake Review")