# Resumo do Notebook *01-regex.ipynb*

## 1. Leitura e Análise do Arquivo
Este notebook aborda o uso de Expressões Regulares (Regex) em Python, apresentando conceitos fundamentais de correspondência de padrões de texto e aplicações práticas. Adicionalmente, podem haver menções e exemplos envolvendo bibliotecas como **NumPy**, **Matplotlib** e **scikit-learn** (sklearn) para manipulação de dados, visualização e possíveis aplicações em aprendizado de máquina.

## 2. Estruturação do Resumo
Para facilitar a compreensão, este resumo está dividido nas seguintes seções:

1. **Introdução às Expressões Regulares**  
2. **Principais Funções e Metacaracteres**  
3. **Exemplos de Código e Explicações Linha a Linha**  
4. **Uso de Bibliotecas Complementares**  
   - NumPy  
   - Matplotlib  
   - scikit-learn

## 3. Conteúdo do Resumo

### 3.1 Introdução às Expressões Regulares
- **Conceito**: Expressões Regulares (Regex) são padrões utilizados para pesquisar e manipular sequências de caracteres de forma eficiente.
- **Biblioteca `re`**: Em Python, a biblioteca padrão para Regex é a `re`. Ela oferece funções como `match`, `search`, `findall`, `sub`, entre outras, para trabalhar com padrões em strings.

#### Principais Aplicações
- **Validação** de formatos de e-mail, CEP, senhas etc.
- **Extração** de partes específicas de um texto (por exemplo, números em uma frase).
- **Substituição** de padrões em blocos de texto.

### 3.2 Principais Funções e Metacaracteres

#### Funções mais utilizadas da biblioteca `re`:
1. **`re.match(pattern, string)`**  
   - Tenta casar o padrão (`pattern`) apenas no início da string.
2. **`re.search(pattern, string)`**  
   - Busca pelo padrão em qualquer posição da string.
3. **`re.findall(pattern, string)`**  
   - Retorna uma lista com todos os casamentos encontrados.
4. **`re.split(pattern, string)`**  
   - Separa a string em uma lista de substrings, usando o padrão como delimitador.
5. **`re.sub(pattern, repl, string)`**  
   - Substitui todas as ocorrências do padrão por `repl` na string.

#### Metacaracteres Comuns:
- `.` (ponto): corresponde a qualquer caractere (exceto nova linha).
- `^` (circunflexo): indica início da string.
- `$` (cifrão): indica final da string.
- `*` (asterisco): casa zero ou mais ocorrências do padrão anterior.
- `+` (sinal de mais): casa uma ou mais ocorrências do padrão anterior.
- `?` (interrogação): casa zero ou uma ocorrência do padrão anterior.
- `{m,n}`: casa de `m` até `n` ocorrências do padrão anterior.
- `[]`: define um conjunto de caracteres.
- `|`: operador “OR” (ou).
- `(...)`: agrupa padrões e permite a captura de grupos.

### 3.3 Exemplos de Código e Explicações Linha a Linha

A seguir, alguns trechos de código em Python que exemplificam o uso de Regex. As linhas serão comentadas para demonstrar suas funcionalidades.

```python
import re  # Importa a biblioteca de expressões regulares

# Exemplo 1: Uso de re.search para encontrar um padrão em qualquer posição da string
texto = "Hoje é dia 09/03/2025 e amanhã será 10/03/2025"
padrao_data = r"\d{2}/\d{2}/\d{4}"  # Padrão para datas no formato DD/MM/YYYY

resultado = re.search(padrao_data, texto)
if resultado:
    print("Data encontrada:", resultado.group())
```
Explicações linha a linha:
	1.	import re: Importa a biblioteca re, que contém funções de Regex.
	2.	texto = ...: Cria uma variável de string com uma frase que contém datas.
	3.	padrao_data = r"\d{2}/\d{2}/\d{4}": Define o padrão de data usando a notação raw string (r"") para evitar problemas com barras invertidas. O metacaractere \d indica “dígito”, {2} e {4} especificam o número de repetições.
	4.	resultado = re.search(padrao_data, texto): Procura o primeiro padrão de data dentro da string texto.
	5.	if resultado: ...: Verifica se houve casamento com o padrão e, em caso positivo, imprime a data encontrada.

⸻




```python
import re
# Exemplo 2: Uso de re.findall para obter todas as datas
texto = "As datas são 09/03/2025, 10/03/2025 e 11/03/2025."
padrao_data = r"\d{2}/\d{2}/\d{4}"

todas_as_datas = re.findall(padrao_data, texto)
print("Todas as datas:", todas_as_datas)
```

Explicações linha a linha:
	1.	import re: Reforça a importação da biblioteca (caso não tenha sido feita anteriormente).
	2.	texto = ...: String contendo diversas datas.
	3.	padrao_data = ...: Define o mesmo padrão usado no exemplo anterior.
	4.	todas_as_datas = re.findall(...): Retorna todas as ocorrências de datas no texto em uma lista.
	5.	print(...): Exibe o resultado na tela.

⸻


```python

import re

# Exemplo 3: Substituindo um padrão
texto = "Contato: email_teste@exemplo.com"
padrao_email = r"[\w\.-]+@[\w\.-]+"

texto_anonimizado = re.sub(padrao_email, "[EMAIL_REMOVIDO]", texto)
print("Texto anonimizado:", texto_anonimizado)
```
Explicações linha a linha:
	1.	import re: Importa novamente a biblioteca re.
	2.	texto = ...: Contém um endereço de e-mail.
	3.	padrao_email = r"[\w\.-]+@[\w\.-]+": Define um padrão simples para e-mails (caracteres alfanuméricos, ponto e hífen antes e depois de “@”).
	4.	texto_anonimizado = re.sub(...): Substitui o e-mail por [EMAIL_REMOVIDO].
	5.	print(...): Exibe o texto resultante sem o e-mail original.

⸻

## 3.4 Uso de Bibliotecas Complementares

Embora o foco principal seja o uso de Regex, o notebook pode incluir exemplos de como integrar essas expressões regulares com dados ou fluxos de trabalho mais complexos. Abaixo seguem exemplos hipotéticos de como isso poderia ser feito.

3.4.1 NumPy

Caso o notebook utilize NumPy para manipular arrays e dados numéricos junto das expressões regulares, um exemplo simples poderia ser:

```python

import numpy as np
import re


# Exemplo de aplicação onde combinamos Regex com dados numéricos
lista_textos = np.array(["Valor: 100", "Valor: 200", "Valor: 300"])

padrao_valor = r"\d+"
valores_extraidos = []

for texto in lista_textos:
    match = re.search(padrao_valor, texto)
    if match:
        valores_extraidos.append(int(match.group()))

valores_array = np.array(valores_extraidos)
print("Array de valores:", valores_array)
```

Explicações linha a linha:
	1.	import numpy as np: Importa a biblioteca NumPy.
	2.	import re: Importa a biblioteca de expressões regulares.
	3.	lista_textos = np.array([...]): Cria um array NumPy de strings.
	4.	padrao_valor = r"\d+": Define um padrão para capturar um ou mais dígitos.
	5.	valores_extraidos = []: Lista para armazenar os valores numéricos extraídos.
	6.	Loop for texto in lista_textos: percorre cada string do array.
	7.	match = re.search(...): Procura o primeiro valor numérico na string.
	8.	valores_extraidos.append(int(match.group())): Converte a ocorrência encontrada em inteiro e adiciona à lista.
	9.	valores_array = np.array(valores_extraidos): Converte a lista final em array NumPy.
	10.	print(...): Exibe o array de valores numéricos.

3.4.2 Matplotlib

Para ilustrar resultados, pode-se usar Matplotlib:
```python

import matplotlib.pyplot as plt
import re

texto_grande = "Produto A: R$10, Produto B: R$25, Produto C: R$40"
padrao_preco = r"R\$(\d+)"

precos = re.findall(padrao_preco, texto_grande)
precos = [int(p) for p in precos]

plt.bar(["Produto A", "Produto B", "Produto C"], precos)
plt.title("Preços de Produtos")
plt.xlabel("Produtos")
plt.ylabel("Preço em R$")
plt.show()
```
Explicações linha a linha:
	1.	import matplotlib.pyplot as plt: Importa a biblioteca Matplotlib para gerar gráficos.
	2.	import re: Importa a biblioteca de expressões regulares.
	3.	texto_grande = ...: String contendo informações sobre produtos e preços.
	4.	padrao_preco = r"R\$(\d+)": Padrão que captura apenas os dígitos após “R$”. Os parênteses (...) definem um grupo de captura.
	5.	precos = re.findall(...): Retorna todos os preços encontrados como strings.
	6.	precos = [int(p) for p in precos]: Converte cada preço de string para inteiro.
	7.	plt.bar(...): Cria um gráfico de barras para visualizar os preços.
	8.	plt.title(...), plt.xlabel(...), plt.ylabel(...): Define título e rótulos dos eixos.
	9.	plt.show(): Exibe o gráfico.

3.4.3 scikit-learn (sklearn)

Em alguns casos, o uso de Regex pode aparecer em cenários de limpeza de texto antes de alimentar um modelo de machine learning. Por exemplo:
```python

import re
from sklearn.feature_extraction.text import CountVectorizer

# Exemplo de limpeza básica antes de criar vetor de atributos
textos = ["Eu adoro Python!!!", "Regex é muito útil.", "Machine Learning & Python"]
textos_limpos = []

for t in textos:
    # Remove pontuação usando Regex
    texto_sem_pontuacao = re.sub(r"[^\w\s]", "", t)
    textos_limpos.append(texto_sem_pontuacao.lower())

# Agora utiliza CountVectorizer para criar um vetor de frequência das palavras
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos_limpos)

print("Vocabulário:", vectorizer.vocabulary_)
print("Matriz de Frequências:\n", X.toarray())
```

Explicações linha a linha:
	1.	import re: Importa a biblioteca de expressões regulares.
	2.	from sklearn.feature_extraction.text import CountVectorizer: Função do scikit-learn que converte coleções de texto em matrizes de contagem de tokens.
	3.	textos = [...]: Lista de textos brutos.
	4.	textos_limpos = []: Lista para armazenar textos após limpeza.
	5.	Loop for t in textos: itera sobre cada texto.
	6.	re.sub(r"[^\w\s]", "", t): Remove todos os caracteres que não são alfanuméricos (\w) ou espaços (\s).
	7.	texto_sem_pontuacao.lower(): Converte o texto para minúsculo para uniformizar a análise.
	8.	vectorizer = CountVectorizer(): Cria uma instância do CountVectorizer.
	9.	X = vectorizer.fit_transform(textos_limpos): Ajusta o vetor e transforma os textos em matriz de frequência.
	10.	print("Vocabulário:", ...): Exibe o mapeamento de palavras para índices.
	11.	print("Matriz de Frequências:\n", ...): Exibe a matriz de contagem de cada palavra em cada texto.

⸻

4. Atenção Especial às Bibliotecas
	1.	re (Regex):
	•	Permite operações complexas de busca e substituição em strings.
	•	Funções principais: match, search, findall, sub, split.
	•	Metacaracteres e quantificadores: . ^ $ * + ? { } [ ] ( ) \ |.
	2.	numpy:
	•	Facilita a manipulação de grandes quantidades de dados numéricos, permitindo integrar a busca de padrões textuais à análise estatística.
	3.	matplotlib:
	•	Possibilita visualizações de dados extraídos via Regex (por exemplo, contagem de ocorrências, análises de frequência).
	4.	scikit-learn (sklearn):
	•	A Regex pode ser usada para pré-processamento e limpeza de dados textuais antes de modelos de machine learning.
	•	Ferramentas como CountVectorizer, TfidfVectorizer, entre outras, geralmente se beneficiam de strings já “limpas” e formatadas.

⸻

Conclusão

O notebook 01-regex.ipynb apresenta conceitos fundamentais de Expressões Regulares em Python, evidenciando aplicações práticas na extração, busca, substituição e validação de padrões em textos. Ele também demonstra, possivelmente, como integrar Regex a bibliotecas como NumPy (para manipulação de dados), Matplotlib (para visualização) e scikit-learn (para machine learning), oferecendo uma base sólida para resolver desde problemas simples de formatação até tarefas mais complexas de análise de dados textuais.

Principais pontos resumidos:
	•	Regex: Sintaxe básica, funções do módulo re, metacaracteres e quantificadores.
	•	Integração: Exemplos de como integrar Regex com NumPy, Matplotlib e scikit-learn.
	•	Aplicação Prática: Exemplos de códigos completos que podem ser facilmente copiados e adaptados em atividades e provas.

Este foi o resumo completo dos conteúdos abordados no arquivo 01-regex.ipynb, incluindo exemplos de código em Python com explicações detalhadas de cada linha, além de considerações especiais sobre o uso em conjunto com bibliotecas amplamente utilizadas na ciência de dados e no aprendizado de máquina.

# Exemplos de Regex em Python

Este documento reúne diversos exemplos de aplicação de expressões regulares (Regex) em Python, incluindo validação, extração e substituição de padrões. Cada exemplo vem acompanhado de uma breve explicação.

---

## 1. Exemplos de Código e Explicações Linha a Linha

### 1.1 Exemplo 1: Uso de `re.search` para Encontrar um Padrão

```python
import re  # Importa a biblioteca de expressões regulares

# Define um texto contendo datas
texto = "Hoje é dia 09/03/2025 e amanhã será 10/03/2025"
# Padrão para datas no formato DD/MM/YYYY
padrao_data = r"\d{2}/\d{2}/\d{4}"

# Procura o padrão no texto
resultado = re.search(padrao_data, texto)
if resultado:
    print("Data encontrada:", resultado.group())
```
Explicação:
	•	Importa a biblioteca re.
	•	Define um texto que contém datas.
	•	Cria um padrão para identificar datas no formato DD/MM/YYYY.
	•	Usa re.search para localizar a primeira ocorrência e imprime o resultado.

⸻

1.2 Exemplo 2: Uso de re.findall para Obter Todas as Datas
```python
import re

# Texto contendo várias datas
texto = "As datas são 09/03/2025, 10/03/2025 e 11/03/2025."
# Padrão para identificar datas
padrao_data = r"\d{2}/\d{2}/\d{4}"

# Extrai todas as ocorrências do padrão
todas_as_datas = re.findall(padrao_data, texto)
print("Todas as datas:", todas_as_datas)
```
Explicação:
	•	Importa a biblioteca re.
	•	Define um texto com múltiplas datas.
	•	Cria o padrão para datas.
	•	Usa re.findall para extrair todas as datas e imprime a lista resultante.

⸻

1.3 Exemplo 3: Substituindo um Padrão com re.sub
```python
import re

# Texto contendo um e-mail
texto = "Contato: email_teste@exemplo.com"
# Padrão simples para identificar e-mails
padrao_email = r"[\w\.-]+@[\w\.-]+"

# Substitui o e-mail encontrado por [EMAIL_REMOVIDO]
texto_anonimizado = re.sub(padrao_email, "[EMAIL_REMOVIDO]", texto)
print("Texto anonimizado:", texto_anonimizado)
```
Explicação:
	•	Importa a biblioteca re.
	•	Define um texto com um e-mail.
	•	Cria um padrão para identificar e-mails.
	•	Utiliza re.sub para substituir o e-mail pelo marcador [EMAIL_REMOVIDO] e imprime o texto modificado.

⸻

1.4 Exemplo 4: Combinação de Regex com NumPy
```python
import numpy as np
import re

# Cria um array de strings contendo valores
lista_textos = np.array(["Valor: 100", "Valor: 200", "Valor: 300"])
# Padrão para capturar dígitos
padrao_valor = r"\d+"
valores_extraidos = []

# Itera sobre os textos e extrai os números
for texto in lista_textos:
    match = re.search(padrao_valor, texto)
    if match:
        valores_extraidos.append(int(match.group()))

# Converte a lista para um array NumPy e imprime
valores_array = np.array(valores_extraidos)
print("Array de valores:", valores_array)
```

Explicação:
	•	Importa NumPy e a biblioteca re.
	•	Define um array de strings com valores numéricos.
	•	Cria um padrão para capturar dígitos e extrai os números com re.search.
	•	Converte os valores extraídos em um array NumPy e imprime.

⸻

1.5 Exemplo 5: Visualização com Matplotlib e Regex
```python
import matplotlib.pyplot as plt
import re

# Texto contendo informações de preços
texto_grande = "Produto A: R$10, Produto B: R$25, Produto C: R$40"
# Padrão para capturar os dígitos após "R$"
padrao_preco = r"R\$(\d+)"

# Extrai e converte os preços para inteiros
precos = re.findall(padrao_preco, texto_grande)
precos = [int(p) for p in precos]

# Cria um gráfico de barras com os preços
plt.bar(["Produto A", "Produto B", "Produto C"], precos)
plt.title("Preços de Produtos")
plt.xlabel("Produtos")
plt.ylabel("Preço em R$")
plt.show()
```

Explicação:
	•	Importa Matplotlib e re.
	•	Define um texto com preços e cria um padrão para extrair os valores.
	•	Converte os preços extraídos e gera um gráfico de barras.

⸻

1.6 Exemplo 6: Limpeza de Texto com scikit-learn e Regex
```python
import re
from sklearn.feature_extraction.text import CountVectorizer

# Lista de textos com pontuação
textos = ["Eu adoro Python!!!", "Regex é muito útil.", "Machine Learning & Python"]
textos_limpos = []

# Remove a pontuação e converte para minúsculo
for t in textos:
    texto_sem_pontuacao = re.sub(r"[^\w\s]", "", t)
    textos_limpos.append(texto_sem_pontuacao.lower())

# Cria um vetor de frequência das palavras
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos_limpos)

print("Vocabulário:", vectorizer.vocabulary_)
print("Matriz de Frequências:\n", X.toarray())
```
Explicação:
	•	Importa re e CountVectorizer do scikit-learn.
	•	Remove a pontuação de uma lista de textos e os converte para minúsculo.
	•	Cria um vetor de frequência de palavras e imprime o vocabulário e a matriz resultante.

⸻

2. 20 Exemplos Adicionais de Aplicações de Regex

2.1 Validação de E-mail
```python
import re

email = "usuario@exemplo.com"
padrao_email = r"^[\w\.-]+@[\w\.-]+\.\w+$"  # Valida estrutura completa do e-mail

if re.match(padrao_email, email):
    print("Email válido!")
else:
    print("Email inválido!")
```

Explicação:
	•	Define um e-mail e cria um padrão que verifica a estrutura correta (usuário, @, domínio).

⸻

2.2 Validação de Número de Telefone Brasileiro
```python
import re

telefone = "(11) 91234-5678"
padrao_telefone = r"^$begin:math:text$\\d{2}$end:math:text$\s?\d{4,5}-\d{4}$"  # Valida telefones com DDD e número

if re.match(padrao_telefone, telefone):
    print("Telefone válido!")
else:
    print("Telefone inválido!")
```
Explicação:
	•	Define um telefone no formato brasileiro.
	•	O padrão valida o DDD entre parênteses, seguido de um número de 4 ou 5 dígitos e hífen.

⸻

2.3 Validação de URL
```python
import re

url = "https://www.exemplo.com"
padrao_url = r"^(https?:\/\/)?(www\.)?[\w-]+\.[\w-]+(\.[\w-]+)?(\/\S*)?$"

if re.match(padrao_url, url):
    print("URL válida!")
else:
    print("URL inválida!")
```
Explicação:
	•	Define uma URL e cria um padrão que cobre protocolos opcionais, “www”, domínio e caminhos opcionais.

⸻

2.4 Extração de CPF
```python
import re

texto = "Meu CPF é 123.456.789-10"
padrao_cpf = r"\d{3}\.\d{3}\.\d{3}-\d{2}"

cpf = re.search(padrao_cpf, texto)
if cpf:
    print("CPF encontrado:", cpf.group())
```
Explicação:
	•	Define um texto contendo um CPF e utiliza um padrão para capturar o formato ###.###.###-##.

⸻

2.5 Extração de CNPJ
```python
import re

texto = "O CNPJ da empresa é 12.345.678/0001-90"
padrao_cnpj = r"\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}"

cnpj = re.search(padrao_cnpj, texto)
if cnpj:
    print("CNPJ encontrado:", cnpj.group())
```
Explicação:
	•	Define um texto com um CNPJ e cria um padrão para capturar o formato ##.###.###/####-##.

⸻

2.6 Extração de Todos os Números em uma String
```python
import re

texto = "Temos os números 10, 20, 30 e 40"
numeros = re.findall(r"\d+", texto)
print("Números extraídos:", numeros)
```
Explicação:
	•	Define um texto com vários números.
	•	O padrão \d+ captura uma ou mais ocorrências de dígitos.

⸻

2.7 Extração de Palavras que Começam com Letra Maiúscula
```python
import re

texto = "Hoje é um Dia Especial com Muitas Surpresas"
palavras_maiusculas = re.findall(r"\b[A-Z][a-z]*\b", texto)
print("Palavras iniciadas com maiúscula:", palavras_maiusculas)
```
Explicação:
	•	Define um texto e utiliza um padrão para capturar palavras que começam com letra maiúscula.

⸻

2.8 Remoção de Espaços em Branco Extras
```python
import re

texto = "Este   texto  tem   espaços    extras."
texto_limpo = re.sub(r"\s+", " ", texto)
print("Texto limpo:", texto_limpo)
```
Explicação:
	•	Define um texto com múltiplos espaços e substitui sequências de espaços por um único espaço.

⸻

2.9 Remoção de Caracteres Especiais
```python
import re

texto = "Olá!!! Como você está? #incrível"
texto_limpo = re.sub(r"[^\w\s]", "", texto)
print("Texto sem caracteres especiais:", texto_limpo)
```
Explicação:
	•	Define um texto com pontuações e símbolos, removendo-os para deixar apenas letras, números e espaços.

⸻

2.10 Divisão de String Usando Vírgulas e Espaços
```python
import re

texto = "maçã, banana, laranja, uva"
frutas = re.split(r",\s*", texto)
print("Lista de frutas:", frutas)
```
Explicação:
	•	Define uma string com nomes de frutas separados por vírgulas e espaços, dividindo-a em uma lista.

⸻
```python
2.11 Extração de Hashtags de uma Frase

import re

texto = "Curtindo o dia #sol #praia #verão"
hashtags = re.findall(r"#\w+", texto)
print("Hashtags encontradas:", hashtags)
```
Explicação:
	•	Define um texto com hashtags e extrai todas as palavras iniciadas por “#”.

⸻
```python
2.12 Extração de Menções em Redes Sociais

import re

texto = "Olá @usuario1 e @usuario2, bem-vindos!"
mencoes = re.findall(r"@\w+", texto)
print("Menções encontradas:", mencoes)
```
Explicação:
	•	Define um texto com menções (prefixadas com “@”) e extrai todas elas.

⸻

2.13 Encontrar Palavras Repetidas Consecutivamente
```python
import re

texto = "isso é muito muito bom"
repetidas = re.findall(r"\b(\w+)\s+\1\b", texto)
print("Palavras repetidas:", repetidas)
```
Explicação:
	•	Define um texto e utiliza grupos de captura para identificar palavras que se repetem consecutivamente.

⸻

2.14 Extração de Domínios de URLs
```python
import re

urls = ["https://www.google.com", "http://example.org", "https://sub.domain.com"]
dominios = [re.search(r"https?://(?:www\.)?([^/]+)", url).group(1) for url in urls]
print("Domínios extraídos:", dominios)
```
Explicação:
	•	Define uma lista de URLs e utiliza um padrão para extrair apenas o domínio, ignorando o protocolo e o “www”.

⸻

2.15 Extração de Datas no Formato YYYY-MM-DD
```python
import re

texto = "O evento ocorrerá em 2025-03-10 e terminará em 2025-03-15."
datas = re.findall(r"\d{4}-\d{2}-\d{2}", texto)
print("Datas encontradas:", datas)
```
Explicação:
	•	Define um texto com datas no formato ano-mês-dia e extrai todas elas com um padrão específico.

⸻

2.16 Extração de Horários no Formato HH:MM
```python
import re

texto = "O trem parte às 14:30 e chega às 18:45."
horarios = re.findall(r"\b\d{2}:\d{2}\b", texto)
print("Horários encontrados:", horarios)
```
Explicação:
	•	Define um texto contendo horários e utiliza um padrão para extrair horas e minutos.

⸻

2.17 Validação de Formato de Tempo (HH:MM, 24h)
```python
import re

horario = "23:59"
padrao_horario = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
if re.match(padrao_horario, horario):
    print("Horário válido!")
else:
    print("Horário inválido!")
```
Explicação:
	•	Define um horário e valida se ele está no formato 24h correto, garantindo que horas e minutos estejam nos intervalos válidos.

⸻

2.18 Extração de Palavras que Contêm Números
```python
import re

texto = "O modelo X123 tem 4 rodas e o modelo Y456 tem 6 rodas."
palavras_com_numeros = re.findall(r"\b\w*\d\w*\b", texto)
print("Palavras com números:", palavras_com_numeros)
```
Explicação:
	•	Define um texto com palavras que misturam letras e dígitos e extrai todas que contenham pelo menos um número.

⸻

2.19 Reformatando Datas de DD/MM/YYYY para YYYY-MM-DD
```python
import re

texto = "Data original: 09/03/2025"
texto_formatado = re.sub(r"(\d{2})/(\d{2})/(\d{4})", r"\3-\2-\1", texto)
print("Data formatada:", texto_formatado)
```
Explicação:
	•	Define um texto com data no formato DD/MM/YYYY, captura os componentes e reordena para o formato YYYY-MM-DD.

⸻

2.20 Extração de Palavras com Caracteres Acentuados
```python
import re

texto = "Café, maçã, e jiló são frutas ou legumes?"
palavras = re.findall(r"\b[\wÀ-ÿ]+\b", texto)
print("Palavras extraídas:", palavras)
```
Explicação:
	•	Define um texto com palavras contendo caracteres acentuados e utiliza um padrão que abrange o intervalo de caracteres acentuados para extrair corretamente as palavras.

⸻

Fim do documento.
Você pode copiar todo o conteúdo deste arquivo de uma vez utilizando o botão de “Copy” disponível neste bloco.

