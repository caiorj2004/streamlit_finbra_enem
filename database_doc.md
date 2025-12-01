### Documentação Estruturada do Banco de Dados

## Nome do Dataset

- Nome dos Conjuntos de Dados: Despesas Municipais por Função (FIMBRA) e Candidatos do ENEM.

- Fonte: Data IESB (Conexão via PostgreSQL).

- Tabelas Integradas: fimbra_2023, candidatos_enem, municipio, populacao, unidade_federacao.

## Contexto do Negócio

- Objetivo Principal do Projeto: Analisar e quantificar a associação entre os gastos públicos municipais em diversos setores da sociedade (Funções FIMBRA) e o desempenho educacional de jovens (15-19 anos) no Exame Nacional do Ensino Médio (ENEM).

- Problema de Pesquisa: Em que medida os investimentos públicos municipais em diferentes funções sociais impactam a nota média do ENEM dos jovens residentes, e quais setores (funções) são os preditores mais influentes, após o controle de multicolinearidade e dispersão?

## Modelo Conceitual

O dataset final é o resultado da integração relacional (Join) de múltiplas tabelas, sendo a chave de ligação principal o Código IBGE do Município.

- Agregação: A base do ENEM foi agregada pela mediana das notas por município para obter a variável alvo (nota_media).

- Pivotagem: A base FIMBRA (originalmente em formato Long, com uma linha por função) foi pivotada para o formato Wide, onde cada função social se torna uma feature individual ([Função]_per_capita).

- Merge Final: O DataFrame final (df_analise/df_scaled) é a junção desses dados agregados, garantindo que cada linha represente um único município com a média de desempenho ENEM e todas as 28 despesas per capita.

## Dicionário de Dados

A tabela descreve as principais variáveis após a agregação e feature engineering.
Coluna	Tipo de Dados	Descrição	Valores Válidos / Exemplo
Variável Alvo (Y)			
nota_media	float64	Mediana da média das 4 notas objetivas do ENEM (por município, 15-19 anos).	425.88 a 609.30
Identificador			
cod_ibge	object	Código IBGE do município. Chave única no df_scaled.	Ex: 3550308
Variáveis Preditoras (X) - FIMBRA			
[Função]_per_capita	float64	Despesa anual na Função Social específica (ex: Saúde_per_capita), calculada por habitante (R$/capita).	Ex: 1647.46 (Educação)
scaled_[Função]_per_capita	float64	Variável Final (Padronizada). Despesa per capita após a transformação Quantile e Standard Scaling.	Média ≈0, Desvio Padrão ≈1
numero_habitantes	int64	População municipal utilizada para os cálculos.	Ex: 97885
descricao_conta	object	Nome da Função Social de Despesa (Apenas no Long Format).	Ex: 'Educação', 'Saúde', 'Urbanismo', etc.

## Pré-Processamento

As transformações foram estruturadas para mitigar o data leakage e garantir a robustez dos modelos de regressão:
Etapa	Ferramenta	Objetivo
1. Agregação e Feature Engineering	Pandas .groupby(), .median(), Operação Aritmética	Criação da variável alvo (nota_media mediana) e da métrica despesa_per_capita (por linha de função).
2. Pivotagem (Long → Wide)	Pandas .pivot_table()	Transformar as 28 Funções Sociais em 28 colunas distintas para permitir a regressão multivariada.
3. Tratamento de Ausência (NaN)	.fillna(0)	Assumir despesa zero (0) em uma função social quando o registro é ausente após a pivotagem (necessário para evitar perda de observações).
4. Multicolinearidade (X)	VIF (Variance Inflation Factor), Limite 10.0	Remoção iterativa de features (como Relações Exteriores_per_capita) para garantir que as variáveis preditoras restantes fossem estatisticamente independentes.
5. Padronização Robusta	Quantile Transformer + StandardScaler	Reduzir a influência de outliers e da alta dispersão nos dados FIMBRA, garantindo que as features fiquem na mesma escala (Média 0, DP 1) para a modelagem (Necessário para a estabilidade do OLS e a interpretação do GBR/RFR).
6. Particionamento Estratificado	train_test_split(..., stratify=bins)	Dividir os dados em 70% Treino, 15% Validação e 15% Teste, garantindo que a distribuição da nota_media seja preservada em todos os conjuntos.