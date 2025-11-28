import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import pickle

def evaluate_model():
    df = pd.read_csv("data/processed/dataset.csv")

    with open("models/model.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)

    X = vectorizer.transform(df["text"])
    y = df["label"]
    y_pred = model.predict(X)

    os.makedirs("reports/figures", exist_ok=True)

    # DISTRIBUIÇÃO DAS CLASSES
    plt.figure(figsize=(6,5))
    df["label"].value_counts().plot(kind="bar")
    plt.title("Distribuição das Classes")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig("reports/figures/class_distribution.png", dpi=200)
    plt.clf()

    # MATRIZ DE CONFUSÃO
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(pd.unique(y)))
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=200)
    plt.clf()

    # CURVA ROC
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y).ravel()
    y_score = model.predict_proba(X)[:, 1] if len(lb.classes_) == 2 else None

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_bin, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], "--")
        plt.title("Curva ROC")
        plt.xlabel("Falso Positivo")
        plt.ylabel("Verdadeiro Positivo")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/figures/roc_curve.png", dpi=200)
        plt.clf()
    else:
        print("⚠️ ROC não gerada: mais de 2 classes detectadas.")

    print("\nGráficos gerados na pasta: reports/figures/")
    print("⚠️ Há desbalanceamento na base: muitas avaliações positivas em relação às negativas.\n")

if __name__ == "__main__":
    evaluate_model()
