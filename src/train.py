import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

def train_model():
    df = pd.read_csv("data/processed/dataset.csv")

    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_vec, y)

    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump((vectorizer, model), f)

    print("Modelo treinado")

if __name__ == "__main__":
    train_model()
