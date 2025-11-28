import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import pickle

def evaluate_model():
    # Load processed dataset
    df = pd.read_csv("data/processed/dataset.csv")

    # Load trained model
    with open("models/model.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)

    X = vectorizer.transform(df["text"])
    y = df["label"]
    y_pred = model.predict(X)

    os.makedirs("reports/figures", exist_ok=True)

   
    #  DISTRIBUIÇÃO DAS CLASSES

    plt.figure(figsize=(6,5))
    df["label"].value_counts().plot(kind="bar")
    plt.title("Distribuição das Classes (Positivo vs Negativo)")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig("reports/figures/class_distribution.png", dpi=200)
    plt.clf()

    # 2MATRIZ DE CONFUSÃO
   
    cm = confusion_matrix(y, y_pred, labels=["positive", "negative"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positivo", "Negativo"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=200)
    plt.clf()

    #  CURVA ROC
   
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y).ravel()
    y_score = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y_bin, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], "--")
    plt.title("Curva ROC")
    plt.xlabel("Falsa Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curve.png", dpi=200)
    plt.clf()

    print("Gráficos gerados na pasta: reports/figures/")

    print("\n")

    print("há muito mais avaliações positivas do que negativas. isso gera desbalanceamento que pode influenciar o modelo a priorizar a classe majoritária, nesse caso, comentarios positivos.")


if __name__ == "__main__":
    evaluate_model()
