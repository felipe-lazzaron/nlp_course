# Resumo do Notebook *content-based_search.ipynb*

## 1. Leitura e Análise do Arquivo
Este notebook explora métodos de **busca baseada em conteúdo** (*content-based search*) aplicados a um conjunto de artigos científicos (no caso, resumos do arXiv). O objetivo é, a partir de um texto de resumo (*abstract*) fornecido, identificar artigos semelhantes ou relevantes, usando principalmente:

1. **Busca por palavras-chave (inverted index)**  
2. **TF-IDF** para identificar palavras de destaque em um documento  
3. **Modelagem de Tópicos (LDA)** com métricas de avaliação, como *perplexity* e divergência de Jensen-Shannon, para comparar distribuições de tópicos.  

O notebook ainda discute como encontrar um número ótimo de tópicos via perplexidade, como obter representações de cada documento em termos dos tópicos aprendidos, e como calcular distâncias/semelhanças entre distribuições de tópicos (KL, JS). Também sugere comparar essas abordagens de recomendação, possivelmente integrando **LLMs** (Large Language Models) para melhorar a busca.

## 2. Estruturação do Resumo
Este resumo é dividido nas seguintes partes:

1. **Introdução à busca baseada em conteúdo**  
2. **Exercício 1: Busca por palavra-chave**  
3. **Exercício 2: TF-IDF e identificação de melhores palavras-chave**  
4. **Exercício 3: Modelagem de tópicos (LDA) e escolha de número de tópicos**  
5. **Exercício 4: Distâncias entre distribuições de tópicos (KL, JS)**  
6. **Exercício 5: Comparação entre diferentes métodos de recomendação**  

Cada tópico é apresentado com blocos de código em Python, acompanhados de explicações linha a linha. Quando pertinente, há destaques do uso de **scikit-learn**, **NumPy** e outros pacotes relevantes de Python para análise de dados e NLP.

## 3. Conteúdo do Resumo

### 3.1 Introdução à Busca Baseada em Conteúdo
- **Contexto**: O notebook parte de um cenário onde se deseja encontrar artigos que sejam relevantes a um determinado *abstract*.
- **Dados**: É feito um download de um subconjunto de metadados do arXiv (área de CS), contendo títulos e resumos.
- **Objetivo**: Implementar e comparar técnicas de recomendação por:
  1. **Busca por palavra-chave** via índice invertido.
  2. **Busca baseada em TF-IDF**, para identificar *keywords* mais relevantes.
  3. **Modelagem de tópicos** usando LDA, avaliando distâncias entre distribuições de tópicos.

### 3.2 Exercício 1: Busca por Palavra-Chave (Inverted Index)
O notebook sugere criar um **inverted index** (ou índice invertido) para mapear cada palavra aos documentos onde ela aparece. Assim, quando se procura por termos como “autonomous agents” ou “transfer learning”, é possível retornar rapidamente os artigos que contêm esses termos específicos.

```python
# Exemplo hipotético de construção de índice invertido:
inverted_index = {}

for i, texto in enumerate(df['abstract']):
    words = texto.lower().split()
    for w in set(words):
        if w not in inverted_index:
            inverted_index[w] = []
        inverted_index[w].append(i)

# Exemplo de busca:
search_terms = ["autonomous", "agents"]
docs_found = set(inverted_index[search_terms[0]])
for term in search_terms[1:]:
    docs_found = docs_found.intersection(inverted_index[term])
print("Documentos que contêm todos os termos:", docs_found)
```
Linha a linha:
	1.	inverted_index = {}: Dicionário para armazenar { palavra: [lista de documentos] }.
	2.	Loop em df['abstract']: Percorre cada resumo (texto) e seu índice.
	3.	words = texto.lower().split(): Converte o texto em minúsculas e separa por espaços.
	4.	for w in set(words): Itera nas palavras únicas para evitar duplicados no mesmo doc.
	5.	if w not in inverted_index: ...: Se a palavra não está no dicionário, inicializa com lista vazia.
	6.	inverted_index[w].append(i): Adiciona o índice do documento atual à lista de documentos da palavra.
	7.	search_terms = ["autonomous", "agents"]: Exemplo de termos buscados.
	8.	docs_found = set(...): Converte em set para facilitar interseção (busca de conjunção).
	9.	docs_found = docs_found.intersection(...): Para cada termo adicional, faz a interseção dos documentos que contêm esse termo.
	10.	print(...): Mostra a lista de documentos em que ambos os termos foram encontrados.

⸻

3.3 Exercício 2: TF-IDF e Identificação de Melhores Palavras-Chave

Para encontrar termos mais relevantes de um determinado documento, o notebook sugere o uso de TF-IDF. A ideia é “pontuar” as palavras considerando o quanto são frequentes no documento e o quão raras são no corpus.
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Supondo que df['abstract'] contenha os resumos...
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['abstract'])

# Exemplo de como extrair as palavras de maior TF-IDF em um texto específico (ex.: sample_abstract):
sample_vector = vectorizer.transform([sample_abstract])
feature_names = vectorizer.get_feature_names_out()
sorted_indices = sample_vector.toarray()[0].argsort()[::-1]
top_words = [(feature_names[i], sample_vector[0, i]) for i in sorted_indices[:10]]
print("Principais palavras TF-IDF no sample_abstract:", top_words)
```
Linha a linha:
	1.	from sklearn.feature_extraction.text import TfidfVectorizer: Importa a classe para gerar a matriz TF-IDF.
	2.	vectorizer = TfidfVectorizer(...): Configura stopwords, número de features etc.
	3.	tfidf_matrix = vectorizer.fit_transform(df['abstract']): Gera a matriz TF-IDF para todos os abstracts.
	4.	sample_vector = vectorizer.transform([sample_abstract]): Calcula a pontuação TF-IDF para o texto de interesse.
	5.	feature_names = vectorizer.get_feature_names_out(): Obtém o vocabulário mapeado aos índices das colunas.
	6.	sorted_indices = sample_vector.toarray()[0].argsort()[::-1]: Ordena as colunas (palavras) por TF-IDF em ordem decrescente.
	7.	top_words = ...: Extrai as top 10 palavras.
	8.	print(...): Exibe as palavras de maior TF-IDF e seus valores.

Essas palavras podem então ser usadas em um sistema de busca, possivelmente retornando artigos mais similares.

⸻

3.4 Exercício 3: Modelagem de Tópicos (LDA) e Perplexidade

Para descobrir como agrupar documentos em tópicos, o notebook apresenta Latent Dirichlet Allocation (LDA) do scikit-learn. Um ponto chave é escolher o número de tópicos via perplexidade — quanto menor, melhor.
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

vectorizer = CountVectorizer(stop_words='english', min_df=10, max_df=0.8, max_features=1000)
abstract_vectorized = vectorizer.fit_transform(df['abstract'].sample(10000))

for n_components in tqdm([2, 10, 20, 50, 100]):
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42, n_jobs=-1)
    lda.fit(abstract_vectorized)
    print(f\"Number of components: {n_components}. Perplexity: {lda.perplexity(abstract_vectorized)}\")
```
Linha a linha:
	1.	vectorizer = CountVectorizer(...): Vetoriza os documentos (usando contagem).
	2.	abstract_vectorized = vectorizer.fit_transform(...): Seleciona 10.000 resumos para treinamento e cria matriz esparsa.
	3.	for n_components in ...: Faz loop em vários números de tópicos (2, 10, 20, 50, 100).
	4.	lda = LatentDirichletAllocation(...): Cria um modelo LDA com n_components tópicos.
	5.	lda.fit(abstract_vectorized): Treina o modelo.
	6.	print(...): Exibe a perplexidade sobre o mesmo conjunto para avaliar o quão boa é a modelagem.

A perplexidade tende a diminuir até certo ponto; um valor muito alto de n_components pode gerar overfitting, deixando a perplexidade alta novamente ou se estabilizando.

⸻

3.5 Exercício 4: Distâncias entre Distribuições de Tópicos (KL, Jensen-Shannon)

O notebook revisita o conceito de que a saída do LDA para cada documento é um vetor de probabilidade (distribuição sobre os tópicos). Para comparar quão parecidos dois documentos são em termos de tópicos, podemos usar:
	•	KL Divergence (Kullback-Leibler)
	•	Jensen-Shannon Divergence (JS), forma simétrica baseada na KL.

Há um exemplo com scipy.spatial.distance.jensenshannon para comparar distribuições de tópicos.
```python
from scipy.spatial.distance import jensenshannon

lda = LatentDirichletAllocation(n_components=5, random_state=42, n_jobs=-1)
lda.fit(abstract_vectorized)

topics1 = lda.transform(abstract_vectorized[0,:])
topics2 = lda.transform(abstract_vectorized[1,:])
topics3 = lda.transform(abstract_vectorized[500,:])

print(topics1)
print(topics2)
print(topics3)

print(jensenshannon(topics1.ravel(), topics2.ravel()))
print(jensenshannon(topics1.ravel(), topics3.ravel()))
print(jensenshannon(topics2.ravel(), topics3.ravel()))
```
Comentários:
	•	topics1, topics2, topics3 são vetores 1D com a probabilidade de cada tópico (soma = 1).
	•	jensenshannon(...) retorna um escalar que mede a “distância” entre as distribuições de tópico. Quanto menor, mais similares são os documentos.

⸻

3.6 Exercício 5: Comparação das Abordagens

Finalmente, o notebook propõe comparar:
	1.	Recomendação por palavra-chave (inverted index)
	2.	Recomendação por TF-IDF (buscando palavras destacadas no documento)
	3.	Recomendação por LDA (via semelhança entre distribuições de tópico)

Pergunta-se qual abordagem produz resultados mais relevantes, se combinações de métodos podem ser melhores e se LLMs poderiam contribuir (por exemplo, analisando resumo e retornando citações/links adequados).

⸻

4. Observações Importantes
	•	Integração com scikit-learn:
	•	CountVectorizer, TfidfVectorizer e LatentDirichletAllocation são centrais para o pré-processamento e modelagem de tópicos.
	•	NumPy: manipulação de matrizes (ex.: transformando resultados em arrays e ordenando).
	•	Visualização: embora não exista muita plotagem aqui, técnicas de visualização de perplexidade poderiam ser feitas com Matplotlib ou Seaborn.
	•	LLMs: podem enriquecer a análise, mas é preciso ter cuidado com prompts e custos de processamento.

Conclusão:
O notebook content-based_search.ipynb foca em técnicas de busca e recomendação voltadas para artigos científicos, comparando métodos simples (inverted index, TF-IDF) e avançados (LDA com divergência entre distribuições). Ele convida o leitor a analisar a efetividade de cada método, a escolher um número ótimo de tópicos via perplexidade e a explorar métricas de distância, como KL ou Jensen-Shannon, para medir similaridade de tópicos entre documentos. Além disso, sugere ideias de combinar essas abordagens ou mesmo integrar Large Language Models para potencializar a busca de artigos relevantes.

