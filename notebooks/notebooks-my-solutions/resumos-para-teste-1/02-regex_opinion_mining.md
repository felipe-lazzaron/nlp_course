# Resumo do Notebook *02-regex_opinion_mining.ipynb*

## 1. Leitura e Análise do Arquivo

O notebook aborda o tema de **opinion mining** (mineração de opiniões) aplicado a textos de notícias, usando métodos de processamento de linguagem natural (NLP). Ele traz exemplos de como:

1. **Filtrar e pré-processar textos** usando *regular expressions (regex)* e recursos do *Pandas*.
2. **Calcular frequências de documentos** e criar um **dicionário de vocabulário** (document frequency, DF).
3. **Remover stopwords** (palavras muito frequentes e pouco informativas) e aplicar técnicas de **stemming** e **lemmatization** para normalizar palavras.
4. **Construir um índice invertido** para facilitar buscas textuais.
5. **Modelar tópicos** (topic modelling) com o **Latent Dirichlet Allocation (LDA)** oferecido pelo *scikit-learn*.
6. **Combinar** métodos clássicos de NLP com **Large Language Models (LLMs)**, exemplificando como usar prompt engineering para gerar resumos mais avançados.

Ao longo do notebook, são usadas principalmente as bibliotecas:

- **Pandas** para manipulação de dados tabulares.
- **Regex (`re`)** para filtrar e encontrar padrões textuais.
- **NumPy** para operações vetoriais.
- **Matplotlib** para visualização simples de dados.
- **scikit-learn** (sklearn) para vetorização de textos e modelagem de tópicos.
- Uma referência a **NLTK** para *stemming* e *lemmatization*.
- **Google Generative AI** (chamado de “Gemini” no código) para demonstração do uso de **LLMs**.

## 2. Estruturação do Resumo

Este documento está organizado da seguinte forma:

1. **Introdução aos Conceitos do Notebook**  
   (Arquivo, tema de opinion mining e objetivos gerais)

2. **Seções de Código e Explicações Linha a Linha**  
   - Download e leitura dos dados
   - Filtragem por expressões regulares
   - Pré-processamento de textos (ex.: `stopwords`, `lemmatization`, etc.)
   - Construção de vocabulário e DF (document frequency)
   - Índice invertido
   - Modelagem de tópicos com LDA
   - Uso de *Large Language Models* e prompts

3. **Observações Finais**  
   - Resumo do fluxo de opinião mining
   - Destaques de integrações com *NumPy*, *Matplotlib* e *scikit-learn*

## 3. Conteúdo do Resumo

### 3.1 Download dos Dados e Leitura Inicial

No começo do notebook, é feito o download de um dataset de notícias da BBC, contendo milhares de linhas com título, data, link e descrição das notícias.

```python
import kagglehub
import os
import pandas as pd
from pathlib import Path

path = Path(kagglehub.dataset_download("gpreda/bbc-news"))
print("Path to dataset files:", path)
print("Files in the dataset:", os.listdir(path))
df = pd.read_csv(path / os.listdir(path)[0])
print(f"Number of news: {len(df)}")
df.head()
```
Explicação (linha a linha):
	1.	import kagglehub, os, pandas as pd, from pathlib import Path: bibliotecas para download, sistema de arquivos e manipulação de dados.
	2.	path = Path(kagglehub.dataset_download("gpreda/bbc-news")): faz o download do dataset do Kaggle e armazena o caminho local.
	3.	os.listdir(path): lista arquivos no diretório do dataset.
	4.	df = pd.read_csv(...): carrega o primeiro .csv do dataset em um DataFrame do Pandas.
	5.	print(...): exibe o número de linhas no dataset.
	6.	df.head(): mostra as primeiras linhas para inspeção.

Conceito: temos agora cerca de 42.000+ notícias, cada uma com title, pubDate, guid, link e description.

⸻

3.2 Filtragem via Regex (Procurando Brasil, Ukraine, etc.)

O notebook demonstra como filtrar notícias que contêm determinadas palavras, usando expressões regulares (re). Neste exemplo, inicial usa-se algo como r'Brazil*' (apesar de poder melhorar essa regex). Em Pandas, o str.contains() ajuda a identificar linhas cujo campo description contém o padrão.
```python
import re
regular_expression = r'Brazil*'  # regex para encontrar referencias a 'Brazil'
df_filt = df[df['description'].str.contains(regular_expression, case=False)]
print(len(df_filt))
df_filt.head()
```
Explicação:
	1.	import re: biblioteca nativa de expressões regulares em Python.
	2.	regular_expression = r'Brazil*': define um padrão simples (seria mais robusto algo como r'Brazil(ian)?').
	3.	df['description'].str.contains(...): retorna valores booleanos indicando se cada descrição contém o padrão.
	4.	df_filt = df[...]: filtra o DataFrame principal para apenas as linhas que deram True.
	5.	df_filt.head(): inspeciona os resultados.

⸻

3.3 Document Frequency (DF) e Vocabulário

O notebook então concatena todos os textos filtrados e procura todas as palavras, analisando a “document frequency” (quantas vezes cada palavra aparece em documentos distintos). Usa-se a classe CountVectorizer do scikit-learn, definindo binary=True para indicar que importamos apenas se a palavra apareceu ou não (sem contagem multiplicada).
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df_filt['description'])

doc_freq_matrix = X.mean(axis=0)
doc_freq = {
    word: doc_freq_matrix[0, vectorizer.vocabulary_[word]]
    for word in vectorizer.vocabulary_
}
```
Passo a passo:
	1.	from sklearn.feature_extraction.text import CountVectorizer: importamos a ferramenta para transformar textos em matrizes de ocorrência de palavras.
	2.	vectorizer = CountVectorizer(binary=True): cria um vetorizador que converte textos em matrizes de 0/1 (palavra apareceu ou não).
	3.	X = vectorizer.fit_transform(df_filt['description']): ajusta o vocabulário ao texto filtrado e retorna a matriz esparsa X. Cada linha é um documento, cada coluna é uma palavra, e o valor é 1 ou 0.
	4.	doc_freq_matrix = X.mean(axis=0): tira a média por coluna, resultando na fração de documentos que contêm cada palavra (DF).
	5.	doc_freq = {...}: cria um dicionário palavra -> frequência_de_documentos.

⸻

3.4 Stopwords e Visualização com Matplotlib

Ao usar CountVectorizer(stop_words='english'), removemos palavras muito frequentes da língua inglesa que não trazem significado útil (ex.: “the”, “a”, “about”, etc.). Para visualizar, usamos o Pandas e o Matplotlib:
```python
import matplotlib.pyplot as plt
import pandas as pd

df_vocabulary = pd.DataFrame(doc_freq.items(), columns=['word', 'frequency'])
df_vocabulary = df_vocabulary.sort_values(by='frequency', ascending=False)
df_vocabulary.head()

plt.figure(figsize=(15,2))
plt.bar(df_vocabulary.iloc[0:25]['word'], df_vocabulary[0:25]['frequency'])
plt.xticks(rotation=90)
plt.show()
```
Comentário:
	•	df_vocabulary = pd.DataFrame(...): cria DataFrame com duas colunas: word e frequency.
	•	sort_values(...): ordena pela frequência em ordem decrescente.
	•	plt.bar(...): plota as 25 palavras mais frequentes (em termos de DF).
	•	plt.xticks(rotation=90): rotaciona os rótulos do eixo X para legibilidade.

Esta etapa ilustra a integração de Pandas com Matplotlib para análise de frequências de palavras.

⸻

3.5 Stemming e Lemmatization com NLTK

Para melhorar a análise, podemos agrupar palavras de formatos similares (ex.: “says” e “saying”) em um único radical ou “lemma”. O notebook mostra o uso de NLTK para isso:
```python
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
lemmatized_word = lemmatizer.lemmatize("says")

stemmed = PorterStemmer()
stemmed_word = stemmed.stem("says")
print(lemmatized_word, stemmed_word)
```
Passo a passo:
	1.	import nltk: biblioteca principal de NLP em Python.
	2.	from nltk.stem import ...: importamos classes específicas de stemming e lematização.
	3.	nltk.download('wordnet'): obtém dados necessários ao WordNetLemmatizer.
	4.	lemmatizer.lemmatize("says"): retorna “say” (forma de dicionário).
	5.	stemmed.stem("says"): retorna “say” pelo Porter Stemmer.
	6.	print(...): exibe o resultado.

O notebook sugere criar funções que apliquem o stemming ou lemmatization a cada palavra de cada documento, depois recalcular a DF.

⸻

3.6 Índice Invertido (Inverted Index)

Para buscas rápidas, cria-se um inverted index, que mapeia cada palavra para a lista de documentos em que aparece. Exemplo:
```python
def make_inverted_index_from_df(df : pd.DataFrame) -> Dict:
    return {}
```
A ideia seria:
	•	Extrair o vocabulário (palavras únicas).
	•	Para cada palavra, identificar os índices dos documentos onde ela aparece.
	•	Assim, podemos buscar “brazil” AND “president” e rapidamente obter a interseção das listas de documentos.

⸻

3.7 Modelagem de Tópicos (LDA)

O notebook demonstra Latent Dirichlet Allocation usando LatentDirichletAllocation do scikit-learn. É um método de topic modelling que tenta descobrir T tópicos latentes numa coleção de textos. Cada tópico é um conjunto de palavras com pesos.
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=2,
    )),
    ('lda',
     LatentDirichletAllocation(
         n_components=10,
         max_iter=300,
         random_state=42,
     )),
])

pipeline.fit(df_filt['description'])
lda_output = pipeline.transform(df_filt['description'])
lda_model = pipeline.named_steps['lda']
feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()

most_prominent_topic = lda_output.argmax(axis=1)
for topic_idx, topic in enumerate(lda_model.components_):
    num_documents = (most_prominent_topic == topic_idx).sum()
    print(f\"Topic {topic_idx} ({num_documents} documents):\")
    print(\" \".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))
```
Detalhes:
	1.	Pipeline([...]): cria um pipeline com dois passos:
	•	CountVectorizer(...): transforma cada texto em uma contagem de palavras, removendo stopwords e restringindo max_df e min_df.
	•	LatentDirichletAllocation(...): aplica LDA com 10 tópicos (n_components=10).
	2.	pipeline.fit(...): ajusta o pipeline aos textos.
	3.	lda_output = pipeline.transform(...): retorna a distribuição de cada documento sobre os 10 tópicos.
	4.	lda_model = pipeline.named_steps['lda']: obtém o modelo LDA ajustado.
	5.	feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out(): acessa o vocabulário.
	6.	most_prominent_topic = lda_output.argmax(axis=1): identifica o tópico mais provável para cada documento.
	7.	Loop para exibir as 10 palavras mais relevantes (topic.argsort()[:-11:-1]) de cada tópico, e quantos documentos pertencem principalmente a esse tópico.

Este método combina CountVectorizer (do sklearn) e LatentDirichletAllocation para agrupar notícias em tópicos.

⸻

3.8 Large Language Models (LLMs)

Finalmente, o notebook mostra o uso de um LLM (Google Gemini) para criar resumos mais avançados do conjunto de notícias. Depois de concatenar as descrições dos documentos filtrados, passamos isso para o modelo via prompt e pedimos um resumo:
```python
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

news = '\\n\\nAnother news '.join(list(df_filt['description']))
prompt = f\"I have these pieces of news: {news}. Can you summarize them for me?\"

response = model.generate_content(prompt)
print(response.text)
```
Explicação:
	1.	import google.generativeai as genai: biblioteca para acessar a API da Google.
	2.	genai.configure(api_key=GEMINI_API_KEY): configura a chave de acesso (que deve ser guardada com segurança, por exemplo em .env).
	3.	model = genai.GenerativeModel(model_name=\"gemini-1.5-flash\"): seleciona o modelo de LLM.
	4.	news = '\\n\\nAnother news '.join(...): concatena as notícias em um só texto.
	5.	prompt = ...: define a pergunta/pedido para o LLM.
	6.	response = model.generate_content(prompt): obtém a resposta de texto.
	7.	print(response.text): exibe o resumo gerado pelo LLM.

O notebook pergunta sobre diferenças de abordagem (LLM x LDA), e como o prompt engineering pode influenciar o resultado final (mais ou menos verboso, perguntas diretas, etc.).

⸻

4. Observações Finais

Este notebook demonstra diversos aspectos de opinion mining:
	1.	Filtragem Inicial: Aplicação de Regex para isolar subconjuntos de textos que contêm certas palavras de interesse.
	2.	Pré-processamento: Uso de stopwords, análise de frequência de documentos, stemming e lemmatization para normalizar o vocabulário.
	3.	Índice Invertido: Estratégia para buscas rápidas de termos em grandes coleções de texto.
	4.	Modelagem de Tópicos: Latent Dirichlet Allocation com scikit-learn para agrupar documentos em tópicos latentes.
	5.	Integração com LLMs: Utilização de API (Google Gemni) para criar resumos e/ou responder perguntas de forma mais sofisticada do que métodos estritamente baseados em contagem de palavras.

Uso de Bibliotecas em Destaque
	•	Regex (re): Essencial para filtrar e manipular texto com padrões específicos.
	•	NumPy: Manipulação de matrizes e arrays, inclusive matrizes esparsas resultantes de vetorização.
	•	Matplotlib: Visualização de frequências e resultados de pré-processamento.
	•	scikit-learn:
	•	CountVectorizer para criar matrizes de termos (bag-of-words).
	•	LatentDirichletAllocation para agrupar documentos em tópicos.
	•	Pipeline para organizar etapas de processamento.
	•	NLTK: Stemming e lemmatization para normalizar formas de palavras.
	•	Google Generative AI (Gemini): Demonstra como LLMs podem ser aplicados para resumos e análises qualitativas de grandes coleções de texto.

⸻

Conclusão:
O arquivo 02-regex_opinion_mining.ipynb exibe, na prática, técnicas importantes de NLP para mineração de opinião, desde filtragem e análise de frequência até modelagem de tópicos. Por fim, mostra como LLMs podem complementar esses processos clássicos, oferecendo resumos contextuais e respostas mais interpretativas para grandes coleções de texto.

