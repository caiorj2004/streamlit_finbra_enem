# data_processing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import numpy as np
import os
from data_ingestion import carregar_e_combinar_dados 

# --- CONFIGURAÇÃO DE VARIÁVEIS CHAVE ---
ID_COL = 'cod_ibge' 
NOTA_ALVO = 'nota_media'
FEATURE_VALUE_COL = 'despesa_per_capita'
FEATURE_NAME_COL = 'descricao_conta'
COLUNAS_ESSENCIAIS = ['Saúde_per_capita', 'Educação_per_capita'] 
NOME_ARQUIVO_PREPROCESSOR = 'models/preprocessor_fimbra_scaled.pkl' 
# NOVOS ARTEFATOS PARA O DASHBOARD (EDA)
NOME_ARQUIVO_DF_LONG_EDA = 'models/df_eda_long_format.pkl'
NOME_ARQUIVO_DF_FILTERS_EDA = 'models/df_eda_filters.pkl'


# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO E ESCALAMENTO
# ----------------------------------------------------------------------

def processar_e_escalar_dados(df_long, df_filters): # AGORA ACEITA DOIS ARGS
    """
    Consome o DataFrame Long (df_long) e o DataFrame de Filtros (df_filters),
    salva ambos para o dashboard, e então realiza a pivotagem, VIF, 
    padronização e serializa o objeto preprocessor.
    Retorna o DataFrame Wide/Scaled e a lista final de features.
    """
    
    # --- 1. SERIALIZAÇÃO RÁPIDA PARA O DASHBOARD (EDA) ---
    print("\n--- SERIALIZANDO DATAFRAMES PARA O DASHBOARD (EDA) ---")
    df_long.to_pickle(NOME_ARQUIVO_DF_LONG_EDA)
    df_filters.to_pickle(NOME_ARQUIVO_DF_FILTERS_EDA)
    print(f"✅ Arquivos de EDA (.pkl) salvos em '{os.path.dirname(NOME_ARQUIVO_PREPROCESSOR)}/'")
    
    # --- 2. CONFIGURAÇÃO E EXTRAÇÃO DE FUNÇÕES ---
    FUNCOES_SOCIAIS_COMPLETA = df_long[FEATURE_NAME_COL].unique().tolist()
    FUNCOES_SOCIAIS_COMPLETA = [f for f in FUNCOES_SOCIAIS_COMPLETA if isinstance(f, str) and f.strip() != ''] 

    # ----------------------------------------------------------------------
    # 3. PIVOTAGEM DE DADOS (USANDO TODAS AS FUNÇÕES)
    # ----------------------------------------------------------------------

    print(f"\n--- INICIANDO PIVOTAGEM DE DADOS ({len(FUNCOES_SOCIAIS_COMPLETA)} FUNÇÕES) ---")

    df_filtered = df_long[df_long[FEATURE_NAME_COL].isin(FUNCOES_SOCIAIS_COMPLETA)].copy()

    df_wide_features = df_filtered.pivot_table(
        index=ID_COL, 
        columns=FEATURE_NAME_COL, 
        values=FEATURE_VALUE_COL
    ).reset_index()

    new_cols = {col: f'{col}_per_capita' for col in FUNCOES_SOCIAIS_COMPLETA if col in df_wide_features.columns}
    df_wide_features.rename(columns=new_cols, inplace=True)

    FEATURE_COLS_WIDE = [f'{func}_per_capita' for func in FUNCOES_SOCIAIS_COMPLETA if f'{func}_per_capita' in df_wide_features.columns]
    df_wide_features[FEATURE_COLS_WIDE] = df_wide_features[FEATURE_COLS_WIDE].fillna(0)

    df_nota_unica = df_long[[ID_COL, NOTA_ALVO]].drop_duplicates(subset=[ID_COL]).set_index(ID_COL)
    df_analise = df_wide_features.set_index(ID_COL).join(df_nota_unica, how='inner').reset_index()

    df_analise.dropna(subset=[NOTA_ALVO], inplace=True) 

    print(f"DataFrame Analítico (Wide Format) criado com {df_analise.shape[0]} observações (Pós fillna(0)).")
    print(f"Total de Features Wide: {len(FEATURE_COLS_WIDE)}")
    print("-" * 50)


    # ----------------------------------------------------------------------
    # 4. TRATAMENTO DE MULTICOLINEARIDADE (VIF) - COM PROTEÇÃO DE VARIÁVEIS CHAVE
    # ----------------------------------------------------------------------
    
    # ... (O loop VIF continua inalterado) ...
    # O código de VIF aqui é apenas a seção VIF do seu bloco original.
    
    X_vif = df_analise[FEATURE_COLS_WIDE].copy()
    threshold_vif = 10.0
    features_finais = FEATURE_COLS_WIDE.copy()

    while True:
        if X_vif.shape[1] < 2: break
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        try:
            vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        except np.linalg.LinAlgError:
            break
        max_vif = vif_data['VIF'].max()
        if max_vif > threshold_vif:
            drop_index = vif_data['VIF'].idxmax()
            feature_to_drop = vif_data.loc[drop_index, 'feature']
            if feature_to_drop in COLUNAS_ESSENCIAIS and len(X_vif.columns) > 2:
                vif_data_sorted = vif_data.sort_values(by='VIF', ascending=False)
                feature_to_drop_safe = feature_to_drop
                for feat in vif_data_sorted['feature']:
                    if feat not in COLUNAS_ESSENCIAIS:
                        feature_to_drop_safe = feat
                        break
                if feature_to_drop_safe in COLUNAS_ESSENCIAIS:
                     X_vif = X_vif.drop(columns=[feature_to_drop])
                     features_finais.remove(feature_to_drop)
                else: 
                     X_vif = X_vif.drop(columns=[feature_to_drop_safe])
                     features_finais.remove(feature_to_drop_safe)
            else: 
                X_vif = X_vif.drop(columns=[feature_to_drop])
                features_finais.remove(feature_to_drop)
        else:
            print(f"VIF OK. Máximo VIF: {max_vif:.2f}. {len(features_finais)} features mantidas.")
            break
    print(f"\nFEATURES FINAIS (após VIF < 10.0): {len(features_finais)} colunas mantidas.")


    # ----------------------------------------------------------------------
    # 5. PRÉ-PROCESSAMENTO ROBUSTO (PADRONIZAÇÃO) E SERIALIZAÇÃO
    # ----------------------------------------------------------------------

    print("\n--- INICIANDO PADRONIZAÇÃO ROBUSTA E SERIALIZAÇÃO ---")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 
             Pipeline(steps=[
                 ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=df_analise.shape[0], random_state=42)),
                 ('scaler', StandardScaler()) 
             ]), 
             features_finais)
        ],
        remainder='passthrough' 
    )

    cols_processadas = features_finais + [ID_COL, NOTA_ALVO]
    df_temp = df_analise[cols_processadas].copy()

    X_transformed = preprocessor.fit_transform(df_temp)

    transformed_feature_names = [f'scaled_{col}' for col in features_finais]
    final_cols = transformed_feature_names + [ID_COL, NOTA_ALVO] 

    df_scaled = pd.DataFrame(X_transformed, columns=final_cols, index=df_temp.index)

    # 5.4. Serializa o Preprocessor
    joblib.dump(preprocessor, NOME_ARQUIVO_PREPROCESSOR)

    print(f"✅ Pré-processador serializado em '{NOME_ARQUIVO_PREPROCESSOR}'")

    return df_scaled, features_finais

# ----------------------------------------------------------------------
# LÓGICA DE EXECUÇÃO PARA REUSO DO MÓDULO (MLOps)
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # Cria o diretório 'models/' se não existir para serialização
    if not os.path.exists('models'): os.makedirs('models')
        
    # ATENÇÃO: data_ingestion.py deve retornar df_long E df_filters agora!
    df_long, df_filters = carregar_e_combinar_dados() 
    df_scaled, features_finais = processar_e_escalar_dados(df_long, df_filters)
    print(f"Processamento e escalamento concluídos. df_scaled shape: {df_scaled.shape}")