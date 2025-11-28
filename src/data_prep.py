import pandas as pd
import os

def prepare_dataset():
    input_path = "data/raw/dataset.csv"
    output_path = "data/processed/dataset.csv"

    df = pd.read_csv(input_path)

    # Usa o texto pré-processado
    df = df[["review_text_processed", "polarity"]].dropna()

    # Renomeia para ficar igual ao seu train.py atual
    df = df.rename(columns={
        "review_text_processed": "text",
        "polarity": "label"
    })

    df["label"] = df["label"].map({
        1.0: "positive",
        0.0: "negative"
    })


    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset pronto: {output_path}")

if __name__ == "__main__":
    prepare_dataset()
