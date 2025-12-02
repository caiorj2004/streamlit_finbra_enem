# app.py

# ----------------------------------------
# 1. Bibliotecas e Configuração
# ----------------------------------------
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os 
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 

# Definir variáveis globais
NOTA_ALVO = 'nota_media'
ID_COL = 'cod_ibge'
FEATURE_VALUE_COL = 'despesa_per_capita'
FEATURE_NAME_COL = 'descricao_conta'
# Nomes dos Artefatos Serializados
NOME_MODELO_SERIALIZADO = 'models/rfr_model.pkl'
NOME_ARQUIVO_DF_LONG_EDA = 'models/df_eda_long_format.pkl'
NOME_ARQUIVO_DF_FILTERS_EDA = 'models/df_eda_filters.pkl'

# Listas de apoio
NOTAS_DISPONIVEIS = ['nota_media', 'nota_ciencias_da_natureza', 'nota_ciencias_humanas', 'nota_linguagens_e_codigos', 'nota_matematica', 'nota_redacao']
FUNCOES_FIMBRA_PADRAO = ['Educação', 'Saúde', 'Urbanismo', 'Assistência Social', 'Cultura', 'Administração', 'Saneamento Básico']


# ----------------------------------------
# 2. CARREGAMENTO RÁPIDO DE ARTEFATOS (Pós-Pipeline)
# ----------------------------------------

@st.cache_data
def carregar_dados_eda():
    """Carrega os DataFrames de EDA e Filtros diretamente do disco (.pkl)."""
    try:
        df_long_loaded = pd.read_pickle(NOME_ARQUIVO_DF_LONG_EDA)
        df_filters_loaded = pd.read_pickle(NOME_ARQUIVO_DF_FILTERS_EDA)
        return df_long_loaded, df_filters_loaded
    
    except FileNotFoundError:
        st.error(
            "ERRO CRÍTICO: Arquivos de dados de EDA não encontrados! "
            "Execute o 'python run_pipeline.py' para gerar os arquivos .pkl na pasta 'models/'."
        )
        return pd.DataFrame(), pd.DataFrame()


@st.cache_resource
def carregar_modelos_serializados(df_dados_brutos):
    """
    Carrega o modelo RFR e RECONSTRÓI o preprocessor no código (solução definitiva
    contra o AttributeError de serialização).
    
    Args:
        df_dados_brutos (pd.DataFrame): O DataFrame Long Format (df_long) contendo 
                                        os dados brutos de EDA.
                                        
    Returns:
        tuple: (model, preprocessor, features_finais_raw)
    """
    # 1. Carregar o Modelo RFR Serializado
    try:
        model = joblib.load(NOME_MODELO_SERIALIZADO)
    except Exception:
        st.error(f"Erro ao carregar o modelo de regressão '{NOME_MODELO_SERIALIZADO}'.")
        return None, None, []

    # 2. DEFINIÇÃO DA LISTA DE FEATURES BRUTAS
    
    # Inferimos as colunas _per_capita que serão ajustadas, assumindo que são as que passaram
    # pelo pipeline, pois o .pkl quebrado não pode ser lido.
    features_finais_raw = [c for c in df_dados_brutos.columns if c.endswith('_per_capita')]
    
    # 3. Reconstruir o ColumnTransformer em código
    
    # Define o pipeline de transformação (replica a lógica do data_processing.py)
    transformador_numerico = Pipeline(steps=[
        # QuantileTransformer: Essencial para mitigar outliers e assimetria.
        # n_quantiles é definido como o tamanho do DF para robustez máxima.
        ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=df_dados_brutos.shape[0], random_state=42)),
        ('scaler', StandardScaler())
    ])
    
    # Cria o ColumnTransformer (o preprocessor)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', transformador_numerico, features_finais_raw) # Features a serem transformadas
        ],
        remainder='passthrough', # Mantém colunas DUMMY (ID_COL, NOTA_ALVO)
        n_jobs=-1
    )
    
    # 4. Ajustar (FIT) o preprocessor aos dados (Solução contra o AttributeError)
    try:
        # Colunas que o preprocessor precisa ver para o FIT: FEATURES_FINAIS_RAW + [ID_COL, NOTA_ALVO]
        cols_para_fit = features_finais_raw + [ID_COL, NOTA_ALVO]
        
        # O fit é feito no DF completo (df_dados_brutos) garantindo que NaNs sejam tratados com 0, 
        # o que é consistente com a etapa fillna(0) antes do fit no pipeline.
        preprocessor.fit(df_dados_brutos[cols_para_fit].fillna(0)) 
        
    except Exception as e:
        st.error(f"Erro CRÍTICO ao REAJUSTAR o preprocessor (FIT). A lista de colunas pode estar incompleta: {e}")
        return model, None, features_finais_raw

    # 5. Retorno Final
    return model, preprocessor, features_finais_raw


# ----------------------------------------
# 3. Funções de Visualização e Manipulação
# ----------------------------------------
# ... (Funções plot_histograma_notas, plot_boxplots_despesas_long, etc. inseridas aqui) ...
@st.cache_data
def criar_df_wide_para_ranking(df_long):
    """
    Cria um DataFrame Wide (Notas + Todas Despesas Per Capita) para rankings rápidos.
    """
    # 1. Lista DINÂMICA de todas as funções FIMBRA disponíveis no DF Long
    funcoes_completas = df_long[FEATURE_NAME_COL].unique().tolist()
    funcoes_completas = [f for f in funcoes_completas if isinstance(f, str) and f.strip() != '']
    
    # 2. Pivotagem das despesas
    df_wide_features = df_long.pivot_table(
        index=ID_COL, 
        columns=FEATURE_NAME_COL, 
        values=FEATURE_VALUE_COL
    ).fillna(0).reset_index()

    # 3. Renomeia e adiciona sufixo _per_capita
    # A lista de colunas que serão renomeadas deve ser baseada no DF Wide pivotado
    cols_to_rename = [c for c in df_wide_features.columns if c not in [ID_COL] + NOTAS_DISPONIVEIS]
    new_cols = {col: f'{col}_per_capita' for col in cols_to_rename}
    df_wide_features.rename(columns=new_cols, inplace=True)
    
    # 4. Merge com as Notas/Detalhes do DF Long (para ter nome/UF/Notas)
    cols_detalhes = [ID_COL, 'nome_municipio', 'sigla_uf', 'faixa_populacao'] + NOTAS_DISPONIVEIS
    df_detalhes_unique = df_long.drop_duplicates(subset=[ID_COL])[cols_detalhes].set_index(ID_COL)
    
    df_wide_ranking = df_wide_features.set_index(ID_COL).join(df_detalhes_unique, how='inner').reset_index()
    
    return df_wide_ranking

def plot_histograma_notas(df, nota_col=NOTA_ALVO):
    fig_hist = px.histogram(df, x=nota_col, nbins=30, marginal="box", title="Distribuição da Nota Média (Mediana)")
    fig_hist.add_vline(x=df[nota_col].mean(), line_dash="dash", line_color="red", annotation_text="Média")
    return fig_hist

def plot_boxplots_despesas_long(df, categorias):
    df_filtrado = df[df[FEATURE_NAME_COL].isin(categorias)].copy()
    df_stats = df_filtrado.groupby(FEATURE_NAME_COL)[FEATURE_VALUE_COL].describe().transpose()
    
    st.markdown("### Estatísticas Descritivas das Despesas Per Capita (R$)")
    st.dataframe(df_stats, use_container_width=True)

    fig_box = px.box(df_filtrado, x=FEATURE_NAME_COL, y=FEATURE_VALUE_COL, color=FEATURE_NAME_COL,
                     title="Distribuição das Despesas Per Capita por Função Social (FIMBRA)",
                     labels={FEATURE_NAME_COL: 'Função Social', FEATURE_VALUE_COL: 'Despesa Per Capita (R$)'},
                     notched=True)
    fig_box.update_layout(xaxis_title="", yaxis_tickformat='$,.0f')
    return fig_box, df_stats 

def plot_heatmap_correlacao_long_to_wide(df_long, categorias, nota_col=NOTA_ALVO):
    df_temp = df_long[df_long[FEATURE_NAME_COL].isin(categorias)].copy()
    df_pivot_data = df_temp[[ID_COL, nota_col, FEATURE_NAME_COL, FEATURE_VALUE_COL]].copy()
    df_wide = df_pivot_data.pivot_table(index=ID_COL, columns=FEATURE_NAME_COL, values=FEATURE_VALUE_COL).fillna(0)
    df_nota_unica = df_temp[[ID_COL, nota_col]].drop_duplicates(subset=[ID_COL]).set_index(ID_COL)
    df_corr_analysis = df_wide.join(df_nota_unica, how='inner').reset_index()
    
    cols_analise = [nota_col] + df_wide.columns.tolist() 
    df_corr = df_corr_analysis.dropna(subset=cols_analise)[cols_analise]
    corr_matrix = df_corr.corr()
    
    fig_heat = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                                        colorscale='RdBu', zmin=-1, zmax=1, texttemplate="%{z:.2f}"))
    fig_heat.update_layout(title='Mapa de Correlação: Despesas Per Capita (Seleção) vs. Nota Média ENEM', height=700)
    
    return fig_heat, corr_matrix


# ----------------------------------------
# 4. Layout Streamlit
# ----------------------------------------
st.set_page_config(layout="wide")
st.title("Análise FIMBRA x ENEM")

# Carrega DataFrames e Modelos
df_long, df_enem_agg = carregar_dados_eda()
model, preprocessor, FEATURES_SCALED_NOMES = carregar_modelos_serializados(df_long) 


# Variáveis de apoio
FEATURES_DO_MODELO = [c.replace('_per_capita', '') for c in FEATURES_SCALED_NOMES] # Nomes SEM sufixo para widgets

tabs = st.tabs(["Apresentação e Contexto", "Análise Exploratória (EDA)", "Modelagem e Predição"])

# -------------------
# 4a. Aba Apresentação (Mantida)
# -------------------
with tabs[0]:
    st.header("Contexto e Metodologia: Gastos Sociais e o Desempenho Escolar")
    
    st.markdown("""
    ### Objetivo e Problema de Pesquisa
    
    Este painel de dados tem como objetivo central analisar a **associação estatística** entre os **gastos públicos municipais em diversas funções sociais** (base FIMBRA 2023) e o **desempenho escolar agregado** de jovens (15-19 anos) no ENEM 2023.
    
    O problema de pesquisa é quantificar em que medida o investimento público em setores mediadores — como Saúde, Saneamento e Assistência Social — impacta a nota média do ENEM, e quais funções, para além da Educação direta, são os preditores mais relevantes.
    """)
    
    st.subheader("Relação Setorial (Saúde e Educação)")
    st.markdown("""
    A literatura de desenvolvimento humano estabelece que a relação entre gastos em saúde e resultados educacionais é **indireta, sutil e complexa**. A eficácia dessa relação foi amplamente discutida em estudos como o de **Mello e Pisu (2009, OECD)**, que analisou o impacto dos gastos setoriais em municípios brasileiros.
    
    O investimento em saúde financia uma infraestrutura que, ao melhorar o estado de saúde geral da população, impacta positivamente a capacidade de aprendizado, a frequência escolar e reduz a repetência. O estado de saúde é um **determinante da performance estudantil**.
    
    * **Limitação do ENEM:** O ENEM reflete fortemente o **capital cultural e econômico** do estudante, conforme estudos de **Moraes (2021)**. As notas são viesadas por fatores socioeconômicos, que atuam como poderosas **variáveis de confusão** e limitam o poder explicativo das variáveis de gasto.
    * **Foco na Eficiência:** Estudos como os de **Diniz et al. (2022)** apontam que o impacto do investimento é determinado pela sua **eficiência e composição**, e não apenas pela magnitude do montante gasto.
    """)
    
    st.subheader("Metodologia MLOps e Tratamento Robusto")
    st.markdown("""
    A análise foi conduzida através de um **Pipeline MLOps** estruturado em três etapas principais (Ingestão, Processamento, Modelagem), garantindo a reprodutibilidade dos resultados.
    
    * **Pré-Processamento de Dados Fiscais:** A alta dispersão dos dados FIMBRA exigiu o uso de métricas **per capita** e um tratamento robusto de *scaling* (**Quantile Transformer** + **VIF**) para mitigar *outliers* e controlar a multicolinearidade entre as 28 funções de despesa.
    * **Modelagem e Amostragem:** Foram treinados modelos de **Regressão Linear (OLS)**, **Gradient Boosting Regressor (GBR)** e **Random Forest Regressor (RFR)**. Para evitar vieses de amostragem, foi utilizada a **Amostragem Estratificada** na divisão dos dados (70% Treino, 15% Validação, 15% Teste).
    """)


# -------------------
# 4b. Aba Análise Exploratória (EDA)
# -------------------
with tabs[1]:
    if not df_long.empty:
        st.header("Análise Exploratória de Dados (EDA)")
        
        df_wide_base = criar_df_wide_para_ranking(df_long)
        
        # 1. Filtros de Agregação (UF e Faixa Populacional)
        st.subheader("Filtros de Agregação")
        col_uf, col_pop = st.columns(2)
        
        ufs = sorted(df_enem_agg['sigla_uf'].dropna().unique())
        uf_selected = col_uf.selectbox("Selecionar UF", options=["Todos"] + list(ufs), key="eda_uf")
        
        faixas_pop = sorted(df_enem_agg['faixa_populacao'].dropna().unique())
        pop_selected = col_pop.selectbox("Selecionar Faixa Populacional", options=["Todas"] + list(faixas_pop), key="eda_pop")
        
        # Aplicar filtros
        df_filter_long = df_long.copy()
        df_filter_wide = df_wide_base.copy()
        
        if uf_selected != "Todos":
            df_filter_long = df_filter_long[df_filter_long['sigla_uf']==uf_selected]
            df_filter_wide = df_filter_wide[df_filter_wide['sigla_uf']==uf_selected]
        if pop_selected != "Todas":
            df_filter_long = df_filter_long[df_filter_long['faixa_populacao']==pop_selected]
            df_filter_wide = df_filter_wide[df_filter_wide['faixa_populacao']==pop_selected]


        st.markdown("---")
        
        # SEÇÃO DE NOTAS E RANKING ENEM
        st.subheader("Análise de Desempenho e Ranking ENEM")
        col_dist, col_rank = st.columns(2)
        
        nota_selecionada = col_dist.selectbox("Selecionar Nota para Distribuição e Ranking", options=NOTAS_DISPONIVEIS, key="nota_alvo_select")
        
        with col_dist:
            # Distribuição (Histograma)
            st.markdown(f"#### Distribuição da Nota: {nota_selecionada.replace('_', ' ').title()}")
            st.write(f"Média Geral: {df_filter_long[nota_selecionada].mean():.2f} | Mediana Geral: {df_filter_long[nota_selecionada].median():.2f}")
            
            fig_hist = plot_histograma_notas(df_filter_long, nota_selecionada) 
            fig_hist.update_layout(title=f"Distribuição da Nota: {nota_selecionada.replace('_', ' ').title()}")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_rank:
            # Bloco Disciplinar de Maior Performance
            st.markdown("#### Bloco Disciplinar de Maior Performance")
            areas_sem_media = [n for n in NOTAS_DISPONIVEIS if n != 'nota_media']
            medianas = df_filter_wide[areas_sem_media].median().sort_values(ascending=False)
            
            st.info(f"O bloco com maior performance é: **{medianas.index[0].replace('nota_', '').replace('_', ' ').title()}** (Mediana: {medianas.iloc[0]:.2f})")

            # Ranking TOP 10 ENEM
            top_bottom_select = st.radio("Selecionar Ranking (ENEM):", ['Maiores Notas', 'Menores Notas'], index=0, horizontal=True, key="rank_enem_type")
            
            if top_bottom_select == 'Maiores Notas':
                top10_enem = df_filter_wide.nlargest(10, nota_selecionada)
                title_top = f"Top 10 Municípios por {nota_selecionada.replace('_',' ').title()} (Maiores)"
            else:
                top10_enem = df_filter_wide.nsmallest(10, nota_selecionada)
                title_top = f"Top 10 Municípios por {nota_selecionada.replace('_',' ').title()} (Menores)"

            fig_bar_top10 = px.bar(top10_enem.sort_values(nota_selecionada, ascending=True), x=nota_selecionada, y='nome_municipio', orientation='h', title=title_top, labels={nota_selecionada:'Nota', 'nome_municipio':'Município'}, color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig_bar_top10, use_container_width=True)


        st.markdown("---")
        
        # SEÇÃO DE DESPESAS FIMBRA E COMPOSIÇÃO
        st.subheader("Análise de Despesas Per Capita (FIMBRA)")
        col_desp_filter, col_desp_rank = st.columns(2)
        
        features_fimbra_wide = [c for c in df_filter_wide.columns if c.endswith('_per_capita')]
        
        desp_selected = col_desp_filter.selectbox("Selecionar Função para Ranking e Distribuição:", options=features_fimbra_wide, index=features_fimbra_wide.index('Educação_per_capita') if 'Educação_per_capita' in features_fimbra_wide else 0, key="fimbra_alvo_select")
        
        with col_desp_filter:
            # Boxplot e Estatísticas da despesa selecionada
            desp_selected_raw_name = desp_selected.replace('_per_capita', '')
            df_filter_desp = df_filter_long[df_filter_long[FEATURE_NAME_COL] == desp_selected_raw_name]
            
            st.markdown(f"#### Distribuição de {desp_selected.replace('_', ' ').title()}")
            st.dataframe(df_filter_desp[FEATURE_VALUE_COL].describe().to_frame().T, use_container_width=True)
            
            fig_desp_hist = px.histogram(df_filter_desp, x=FEATURE_VALUE_COL, nbins=30, marginal="box", title=f"Dispersão de {desp_selected}")
            st.plotly_chart(fig_desp_hist, use_container_width=True)


        with col_desp_rank:
            # Ranking TOP 10 FIMBRA
            top_bottom_desp = st.radio("Selecionar Ranking (FIMBRA):", ['Maiores Gastos', 'Menores Gastos'], index=0, horizontal=True, key="rank_fimbra_type")
            
            df_ranking_desp = df_filter_wide[df_filter_wide[desp_selected] > 0].copy() 

            if top_bottom_desp == 'Maiores Gastos':
                top10_desp = df_ranking_desp.nlargest(10, desp_selected)
                title_top_desp = f"Top 10 Municípios por {desp_selected.replace('_',' ').title()} (Maiores)"
            else:
                top10_desp = df_ranking_desp.nsmallest(10, desp_selected)
                title_top_desp = f"Top 10 Municípios por {desp_selected.replace('_',' ').title()} (Menores)"

            fig_bar_top10_desp = px.bar(top10_desp.sort_values(desp_selected, ascending=True), x=desp_selected, y='nome_municipio', orientation='h', title=title_top_desp, labels={desp_selected:'Despesa R$', 'nome_municipio':'Município'}, color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig_bar_top10_desp, use_container_width=True)
            
            # Composição de Gastos
            st.markdown("#### Composição do Gasto Total") 
            
            df_filter_wide['Soma_Despesas_Totais'] = df_filter_wide[features_fimbra_wide].sum(axis=1)
            soma_total_filtrada = df_filter_wide['Soma_Despesas_Totais'].sum()
            
            composicao = df_filter_wide[features_fimbra_wide].sum(axis=0)
            composicao_perc = (composicao / soma_total_filtrada * 100).sort_values(ascending=False).reset_index()
            composicao_perc.columns = ['Função', 'Percentual']
            
            fig_pie_comp = px.pie(composicao_perc.head(5), names='Função', values='Percentual', title="Top 5 Funções que Mais Consomem o Gasto Total", color_discrete_sequence=px.colors.sequential.Greens_r)
            st.plotly_chart(fig_pie_comp, use_container_width=True)

        st.markdown("---")
        
        # SEÇÃO DE CORRELAÇÃO MULTIVARIADA E BIVARIADA (Atualizada)
        st.subheader("Análise de Correlação e Dispersão")

        col_scatter_viz, col_heatmap_viz = st.columns(2)
        
        features_fimbra_wide_all = [c for c in df_filter_wide.columns if c.endswith('_per_capita')]
        
        with col_scatter_viz:
            st.markdown("#### Dispersão Bivariada (Feature vs. Nota Média)")
            
            eixo_x_scatter = st.selectbox("Selecionar Função:", options=features_fimbra_wide_all, index=features_fimbra_wide_all.index('Educação_per_capita') if 'Educação_per_capita' in features_fimbra_wide_all else 0, key="scatter_feature_select")
            
            corr_value = df_filter_wide[[eixo_x_scatter, NOTA_ALVO]].corr().iloc[0, 1]
            
            fig_scatter = px.scatter(df_filter_wide, x=eixo_x_scatter, y=NOTA_ALVO, opacity=0.6, trendline="ols", hover_data=['nome_municipio', 'sigla_uf'], title=f'Nota Média vs. {eixo_x_scatter.replace("_"," ").title()}', labels={eixo_x_scatter: eixo_x_scatter.replace("_"," ").title(), NOTA_ALVO:'Nota Média ENEM'})
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.info(f"Correlação de Pearson: **r = {corr_value:.3f}**")

        with col_heatmap_viz:
            st.markdown("#### Mapa de Correlação (Visão Geral)")
            
            fig_heat, corr_matrix = plot_heatmap_correlacao_long_to_wide(df_filter_long, FUNCOES_FIMBRA_PADRAO)
            
            st.plotly_chart(fig_heat, use_container_width=True)

    else:
        st.warning("Não foi possível carregar os dados de EDA. Verifique se o pipeline foi executado e se os arquivos .pkl estão na pasta 'models/'.")


# -------------------
# 4c. Aba Modelagem e Predição
# -------------------
with tabs[2]:
    st.header("Modelagem Preditiva e Previsão Interativa")
    
    if model and preprocessor:
        st.success("Modelos RFR e Pré-processador carregados com sucesso.")
        
        # --- DISCUSSÃO DOS MODELOS E CONTEXTUALIZAÇÃO ---
        st.subheader("Resultados Chave do Modelo (RFR Vencedor)")
        
        st.markdown("""
        O modelo **Random Forest Regressor (RFR)** foi selecionado por apresentar o melhor equilíbrio entre ajuste e generalização para a natureza não-linear dos dados de gasto. A modelagem preditiva resultou nas seguintes métricas no conjunto de teste:
        """)
        
        col_r2, col_rmse = st.columns(2)
        col_r2.metric(label="R² Final (Teste)", value="0.4478")
        col_rmse.metric(label="RMSE Final (Teste)", value="23.1978")
        
        st.subheader("Considerações Finais sobre a Associação Estatística")
        st.markdown("""
        A análise evidencia que os modelos fornecem associações estatísticas fracas entre as despesas municipais e o desempenho no ENEM. No entanto, é fundamental contextualizar os resultados:
        
        * **Variáveis de Confusão:** O ENEM reflete não apenas a performance educacional, mas também o **capital socioeconômico** (renda familiar, escolaridade dos pais). Esses fatores atuam como **variáveis de confusão**, amplificando ou mascarando as relações observadas.
        * **Correlações Espúrias:** Municípios com maior capacidade fiscal tendem a apresentar **melhores serviços em geral**, gerando **correlações espúrias** que não representam causalidade direta.
        * **Conclusão:** Os resultados descrevem **associações estatísticas que não devem ser interpretados como relações causais diretas**. A natureza multifatorial do desempenho no ENEM exige análises complementares com controles socioeconômicos adequados para uma compreensão mais profunda.
        """)
        
        st.markdown("---")
        
        # --- Lógica de Previsão Interativa (Input Widgets) ---
        st.subheader("Previsão Interativa: Simulação de Gasto Municipal (26 Funções Sociais)")
        
        # Criar a estrutura de input
        inputs = {}
        todas_as_features_do_modelo = FEATURES_SCALED_NOMES
        input_cols = st.columns(3)
        
        for i, feature_full_name in enumerate(todas_as_features_do_modelo):
            feature_name_sem_sufixo = feature_full_name.replace('_per_capita', '')
            
            # 1. Tentar obter o valor mediano BRUTO do DF Long (Para valor default realista)
            try:
                # O df_long está no formato LONG. Filtramos pelo nome bruto da função
                default_value = float(df_long[df_long[FEATURE_NAME_COL] == feature_name_sem_sufixo][FEATURE_VALUE_COL].median())
            except:
                default_value = 100.0 # Valor padrão
                
            # 2. Criar o widget
            inputs[feature_name_sem_sufixo] = input_cols[i % 3].number_input(
                f"Gasto em {feature_name_sem_sufixo.title()} (R$/capita)",
                min_value=0.0,
                value=default_value,
                key=f"input_{feature_full_name}"
            )
            
        predict_button = st.button("Executar Previsão de Nota Média ENEM")
        
        if predict_button:
            
            # 1. CONSTRUÇÃO ROBUSTA DO DATAFRAME DE INPUT (28 colunas)
            todas_features_com_sufixo = FEATURES_SCALED_NOMES
            data_para_preprocessor = {}
            
            # 1.1. Preenche TODAS as 26 FEATURES BRUTAS (Input Real ou Default Zero)
            for feature_full_name in todas_features_com_sufixo:
                feature_name_sem_sufixo = feature_full_name.replace('_per_capita', '')
                input_valor = inputs.get(feature_name_sem_sufixo, 0.0) 
                data_para_preprocessor[feature_full_name] = [input_valor]
                
            # 1.2. Adicionar as 2 Colunas DUMMY (Passthrough)
            data_para_preprocessor[ID_COL] = [0]      
            data_para_preprocessor[NOTA_ALVO] = [0.0] 
            
            input_df_final = pd.DataFrame(data_para_preprocessor)

            # 2. REORDENAR E TRANSFORMAR
            ORDEM_ESPERADA_PELO_PREPROCESSOR = todas_features_com_sufixo + [ID_COL, NOTA_ALVO]
            input_df_final = input_df_final[ORDEM_ESPERADA_PELO_PREPROCESSOR]
            
            input_transformed = preprocessor.transform(input_df_final) 
            
            # 3. PREVISÃO
            X_predict = input_transformed[:, :-2] 
            prediction = model.predict(X_predict)
            
            st.success(f"A Nota Média ENEM prevista para este perfil de gasto é: **{prediction[0]:.2f}**")
            
        st.markdown("---")
        
        # --- Lógica de Feature Importance ---
        st.subheader("Importância das Variáveis no RFR")
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': FEATURES_SCALED_NOMES,
                'Importância': model.feature_importances_
            }).sort_values(by='Importância', ascending=False)
            
            fig_importance = px.bar(
                importance_df.head(10),
                x='Importância',
                y='Feature',
                orientation='h',
                title="Top 10 Fatores Mais Influentes na Nota Média (Pós-VIF)"
            )
            st.plotly_chart(fig_importance, use_container_width=True)

    else:
        st.warning("Modelos não encontrados. Execute o run_pipeline.py para treinar e serializar os modelos.")

