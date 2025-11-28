# Separador de Avaliações NPS — Classificação de Sentimentos com IA

## Equipe
- Raul Vitor Dos Santos Justino — RA: 2225102789 
- Marcello Henrique Dias de Oliveira — RA: 2225100023  

Turma: 41 | Semestre: 2 | Curso: Ciências da computação | Período: Noturno | Ano: 2025

---

## Problema

Trabalhamos em uma concessionária Yamaha que recebe diariamente um grande volume de avaliações de clientes. Essas avaliações são essenciais para monitorar o desempenho da equipe, identificar pontos de melhoria e priorizar atendimentos críticos.

Como as críticas negativas exigem atenção imediata, torna-se necessário automatizar a separação entre feedbacks positivos e negativos. Este projeto desenvolve uma solução de IA capaz de realizar essa classificação automaticamente.

---

## Abordagem de IA

A solução utiliza técnicas de Processamento de Linguagem Natural (PLN) combinadas com um modelo supervisionado. O processo inclui:

1. Pré-processamento dos textos (limpeza, normalização, tokenização)  
2. Vetorização utilizando TF-IDF  
3. Treinamento de um modelo de classificação  
4. Avaliação com:
   - Matriz de confusão  
   - Curva ROC  
   - Métricas como precisão, recall e F1-score  
5. Pipeline para análise de novas frases em tempo real

---

## Dados Utilizados

Dataset:  
https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets/data

Colunas principais:
- original_index  
- review_text  
- review_text_processed  
- review_text_tokenized  
- polarity  
- rating  
- kfold_polarity  
- kfold_rating  

---

# Como Reproduzir o Projeto

# Obrigatório: 
## Python 
## Git

# Como ter acesso ao projeto 

## Clonando repositório: 
Git clone https://github.com/hyyperjustinnn/PROJ_IA_2025_Turma41_Grupo.git

## Entrando no Projeto:
Cd PROJ_IA_2025_Turma2_Grupo1

# ativar ambiente...
Py -m pip install -r requirements.txt 

# preparando Dataset
python src/data_prep.py

# Treinando o modelo.
python src/train.py

# Iniciando I.A
python src/main.py

## criar graficos
python src/evaluate.py


## Resultados do Projeto

O projeto resultou em uma solução de IA capaz de classificar automaticamente avaliações de clientes entre positivas e negativas, auxiliando a concessionária no monitoramento e priorização de atendimentos. O modelo apresenta bom desempenho geral, especialmente em frases diretas e objetivas. No entanto, ainda não é totalmente assertivo em comentários longos, ambíguos ou com negações indiretas, como “não é bom” ou “não é ruim”. Mesmo assim, a ferramenta já oferece ganho significativo de agilidade e organização no processo de análise de feedbacks.





## Estrutura

(src/, data/, models/, reports/, processed/)