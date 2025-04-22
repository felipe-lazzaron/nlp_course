# Aula: Da Biblioteca Scikit-Learn ao PyTorch

## 1. Conceitos Básicos de Machine Learning  
- **Problemas de Classificação:** saída discreta (ex.: detecção de spam, diagnóstico).  
- **Problemas de Regressão:** saída contínua (ex.: previsão de preço).  

---

## 2. Fluxo de Trabalho em Scikit-Learn

### 2.1 Carregando e Visualizando Dados  
```python
from sklearn.datasets import make_classification

# Gera um conjunto sintético de classificação binária
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=15, n_redundant=5,
                           random_state=42)

2.2 Separando em Treino e Teste

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

2.3 Escolhendo e Treinando o Modelo

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)   # evita warning de convergência
model.fit(X_train, y_train)

2.4 Avaliando o Modelo

from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {acc:.2f}")
print("Matriz de Confusão:\n", cm)
```


⸻

# 3. Fluxo de Trabalho em PyTorch

## 3.1 Instalação e Imports
```python
pip install torch torchvision

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```
## 3.2 Preparando Dataset e DataLoader
```
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = MyDataset(X_train, y_train)
test_ds  = MyDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)
```
## 3.3 Definindo a Rede Neural
```python
class SimpleNet(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet(in_features=20, hidden_size=50, num_classes=2)
```
## 3.4 Função de Custo e Otimizador
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
## 3.5 Loop de Treino
```python
num_epochs = 20
model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Época {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```
## 3.6 Avaliação do Modelo
```python
model.eval()
correct = total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, pred = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (pred == y_batch).sum().item()

print(f"Acurácia PyTorch: {correct/total:.2f}")
```
## 3.7 Visualização do Grafo Computacional (opcional)
```python
from torchviz import make_dot

# Gere o grafo a partir de um batch de treino
sample_X, _ = next(iter(train_loader))
dot = make_dot(model(sample_X), params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('computational_graph')
```


⸻

# 4. Exercícios

Exercício 1 – Scikit-Learn
- Ajuste outro classificador (ex.: RandomForest) e compare acurácias.
- Experimente alterar test_size e observar impacto.

Exercício 2 – PyTorch
- Modifique a arquitetura da rede (adicione camadas ocultas).
- Teste outros otimizadores (Adam, RMSprop) e aprimore hyper‑parameters.

⸻

# 5. Conclusão
- Scikit-Learn facilita prototipagem rápida;
- PyTorch oferece controle total do fluxo de dados e gradientes;
- Transferir o fluxo de trabalho de sklearn para PyTorch envolve:
 1. Converter dados em Tensor e Dataset/DataLoader.
 2. Definir nn.Module com forward.
 3. Implementar loop de treino e avaliação manualmente.

> **Dica para a prova:** entenda bem cada etapa do fluxo de dados e a diferença entre o treinamento “pronto” do Scikit-Learn e o loop manual do PyTorch.

# Redes Neurais de Ponta a Ponta para Processamento de Linguagem Natural (PLN)

## 1. Embeddings de Palavras

Matemática subjacente ao bag‑of‑words
Em bag‑of‑words, cada documento é uma matriz X^{(d)}\in\mathbb{R}^{N\times V}, onde N é número de palavras e V tamanho do vocabulário. Para obter embeddings densos, usamos camadas de embedding que mapeiam índices de palavras em vetores de menor dimensão.

Codificação com nn.Embedding

import torch
import torch.nn as nn

# Parâmetros da camada de embedding
vocab_size    = 100   # tamanho do vocabulário
embedding_dim = 2     # dimensão dos vetores

embedding_layer = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Exemplo de batch de tokens (índices)
tokens = torch.tensor([
    [0, 1, 2, 3],
    [4, 5, 4, 3],
    [5, 4, 3, 2]
])

# Gera os embeddings
embeddings = embedding_layer(tokens)
print(embeddings.shape)  # (3, 4, 2)



⸻

2. Estratégias de Tokenização

# Instalação do SentencePiece
!pip install sentencepiece

import sentencepiece as spm
from io import StringIO

# Treina um tokenizer no formato SentencePiece
input_data = "Seu corpus de texto aqui..."
spm.SentencePieceTrainer.train(
    sentence_iterator=StringIO(input_data),
    model_prefix='my_tokenizer',
    vocab_size=8000
)

# Carrega e testa o tokenizer treinado
sp = spm.SentencePieceProcessor()
sp.load('my_tokenizer.model')

test_sentence   = "Este é um exemplo de tokenização."
encoded_pieces  = sp.encode_as_pieces(test_sentence)
encoded_ids     = sp.encode_as_ids(test_sentence)

print(encoded_pieces)
print(encoded_ids)



⸻

3. Preenchimento (Zero‑padding) e Truncamento
	•	Zero‑padding: adiciona zeros ao final de sequências curtas.
	•	Truncamento: corta sequências longas para comprimento máximo fixo.

# Exemplo simples de padding/truncamento
def pad_sequence(seq, pad_idx, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))

# Utilização no Dataset
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, pad_idx):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.pad_idx    = pad_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode_as_ids(self.texts[idx])
        tokens = pad_sequence(tokens, self.pad_idx, self.max_length)
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

# DataLoader
train_ds = TextDataset(X_train, y_train, sp, max_length=100, pad_idx=0)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)



⸻

4. Até Onde Chegamos?

— Já vimos embeddings, tokenização, padding e truncamento; agora vamos montar o pipeline completo.

⸻

5. Criando um Pipeline com PyTorch

5.1 Definição do Modelo

import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.clf = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)          # (batch, seq_len, emb_dim)
        mean = torch.mean(emb, dim=1)    # média sobre seq_len
        return self.clf(mean)            # saída única

model = SimpleClassifier(vocab_size, embedding_dim)
print(model)

5.2 Carregando e Dividindo os Dados

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    'https://raw.githubusercontent.com/tiagoft/NLP/main/wiki_movie_plots_drama_comedy.csv'
)
X_train, X_test, y_train, y_test = train_test_split(
    df['Plot'], df['Genre'].apply(lambda g: int(g=='Drama')),
    test_size=0.2, random_state=42
)

5.3 Treinamento

from tqdm import tqdm

optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
criterion = nn.BCEWithLogitsLoss()

losses = []
model.train()
for epoch in tqdm(range(200)):
    optimizer.zero_grad()
    tokens = sp.encode_as_ids(X_train.tolist())
    tokens = [pad_sequence(t, 0, 100) for t in tokens]
    tokens = torch.tensor(tokens)
    output = model(tokens).flatten()
    loss   = criterion(output, torch.tensor(y_train.values, dtype=torch.float))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())



⸻

6. Avaliando o Modelo

from sklearn.metrics import accuracy_score, f1_score, classification_report

model.eval()
with torch.no_grad():
    tokens = sp.encode_as_ids(X_test.tolist())
    tokens = [pad_sequence(t, 0, 25) for t in tokens]
    tokens = torch.tensor(tokens)
    logits  = model(tokens).flatten()
    probs   = torch.sigmoid(logits)
    preds   = (probs > 0.5).int().numpy()

print("Accuracy:", accuracy_score(y_test, preds))
print("F1 Score:",  f1_score(y_test, preds, average='macro'))
print(classification_report(y_test, preds))



⸻

7. Alguns Passos de Otimização
	•	Ajustar learning rate, experimentar diferentes otimizadores (SGD, Adam, RMSprop)…
	•	Monitorar perda e métricas durante o treino.

⸻

8. Exercícios
	1.	Visualizing embeddings
	•	Plote os vetores gerados por nn.Embedding em 2D (p.ex., TSNE).
	2.	Further optimization
	•	Experimente outras arquiteturas (camadas extras, funções de ativação).
	3.	Advanced modelling
	•	Substitua a média dos embeddings por uma camada recurrente (LSTM/GRU) ou atenção.

⸻

Dica para a prova: compreenda cada etapa do pipeline “end‑to‑end” em PyTorch — da tokenização ao cálculo de métricas — e saiba justificar a escolha de cada componente.