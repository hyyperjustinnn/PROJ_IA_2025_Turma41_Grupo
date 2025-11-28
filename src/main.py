import pickle



with open("models/model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

    print("------------------------------------------")
    print("Bem vindo ao separador de avaliações NPS")
    print("------------------------------------------")

while True:
    
    print("\n")
    frase = input("Digite uma avaliação do NPS: ")
    print("\n")
    
    X = vectorizer.transform([frase])
    pred = model.predict(X)[0]
    
    print("Sentimento da avaliação:", pred)
    print("\n")
    
