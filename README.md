# streamlit_fimbra_enem

# Projeto Disciplinar: MLOps – Impacto dos Gastos Públicos no Desempenho do ENEM

Este projeto foi desenvolvido como parte do currículo acadêmico com o objetivo de investigar a correlação entre os investimentos públicos municipais (em diversas áreas como Saúde, Educação e Urbanismo) e a performance escolar de jovens de 15 a 19 anos no **ENEM 2023**.

## 🎯 Objetivo

Analisar e quantificar em que medida as despesas anuais por habitante (*per capita*) em diferentes funções sociais influenciam a nota média dos estudantes residentes em cada município brasileiro.

## 🛠️ Tecnologias e Arquitetura

O projeto foi estruturado utilizando princípios de **MLOps**, dividindo a lógica em módulos independentes para garantir escalabilidade e reprodutibilidade:

* **Ingestão (`data_ingestion.py`):** Conexão com banco de dados PostgreSQL para extração e integração das bases **FIMBRA**, **ENEM**, **IBGE** e **População**.
* **Processamento (`data_processing.py`):** ETL avançado incluindo pivotagem de dados (Long para Wide), tratamento de valores ausentes, análise de multicolinearidade (**VIF**) e padronização robusta com `QuantileTransformer`.
* **Modelagem (`modeling.py`):** Treinamento comparativo entre modelos de regressão (**OLS**, **Gradient Boosting** e **Random Forest**) com amostragem estratificada.
* **Pipeline (`run_pipeline.py`):** Orquestração completa do fluxo, desde o dado bruto até a serialização do modelo vencedor.

## 📊 Metodologia de Dados

* **Variável Alvo:** Mediana da nota média do ENEM por município.
* **Variáveis Preditoras:** 28 funções de despesas municipais transformadas em gastos *per capita*.
* **Engenharia de Atributos:** Aplicação de filtro de VIF (limite 10.0) para garantir independência estatística entre os preditores.


## ⚠️ Adendo: O Viés da Variável "Nota do ENEM"

É fundamental ressaltar que a nota do ENEM, utilizada aqui como métrica de performance, é uma variável intrinsecamente enviesada por fatores que extrapolam a gestão pública municipal:

1. **Capital Socioeconômico:** O desempenho é fortemente influenciado pela renda familiar e escolaridade dos pais, atuando como uma variável de confusão.
2. **Infraestrutura Escolar:** A diferença de qualidade entre redes privadas e públicas não é capturada apenas pelo gasto municipal, visto que o Estado e a União também aportam recursos.
3. **Natureza Multivariada:** O gasto em saúde ou educação pode levar anos para refletir em notas de exames, e correlações encontradas podem ser espúrias (municípios ricos tendem a ter bons indicadores em tudo, sem necessariamente haver causalidade direta).
