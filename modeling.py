# modeling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import statsmodels.api as sm
import joblib
import os
# Importa a função do módulo anterior (embora o df_scaled seja idealmente passado)
# from data_processing import processar_e_escalar_dados # Não precisa importar se for executado em cadeia

# --- CONFIGURAÇÃO DE VARIÁVEIS CHAVE ---
NOTA_ALVO = 'nota_media'
RANDOM_STATE = 42
NOME_MODELO_SERIALIZADO = 'models/rfr_model.pkl'

# ----------------------------------------------------------------------
# FUNÇÃO AUXILIAR PARA CALCULAR MÉTRICAS
# ----------------------------------------------------------------------
def calcular_metricas(model, X_set, y_true):
    """Calcula e retorna R², RMSE e MAE."""
    y_pred = model.predict(X_set)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return r2, rmse, mae

# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE MODELAGEM E AVALIAÇÃO (CONSUMINDO DF PROCESSADO)
# ----------------------------------------------------------------------

def treinar_e_avaliar_modelos(df_scaled, features_finais): # CORRIGIDO: usa o nome correto
    """
    Divide os dados (70/15/15), treina OLS, GBR, RFR e serializa o modelo vencedor (RFR).
    """
    
    # CORREÇÃO CRÍTICA INTERNA: Usa o nome 'features_finais'
    # features_finais é a lista de nomes RAW (Ex: ['Saúde_per_capita', ...])
    transformed_feature_names_scaled = [f'scaled_{col}' for col in features_finais]
    
    # 1. DEFINIÇÃO DE X E Y
    # X agora usa os nomes 'scaled_'
    X = df_scaled[transformed_feature_names_scaled] 
    y = df_scaled[NOTA_ALVO]

    # 2. CRIAÇÃO DE BINS PARA ESTRATIFICAÇÃO (Para variável contínua)
    df_estratifica = pd.DataFrame({NOTA_ALVO: y})
    df_estratifica['nota_faixa'] = pd.cut(y, bins=10, labels=False, include_lowest=True)
    stratify_target = df_estratifica['nota_faixa']

    print(f"\n--- INICIANDO AMOSTRAGEM ESTRATIFICADA ({X.shape[1]} FEATURES) ---")

    # 3. DIVISÃO: TREINO (70%) e TEMP (30%) - Estratificada
    X_train, X_temp, y_train, y_temp, stratify_train, stratify_temp = train_test_split(
        X, y, stratify_target, test_size=0.3, random_state=RANDOM_STATE, stratify=stratify_target
    )

    # 4. DIVISÃO: VALIDAÇÃO (15%) e TESTE (15%) - Estratificada
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=stratify_temp
    )

    print(f"Treino (70%): {X_train.shape[0]} amostras | Validação (15%): {X_val.shape[0]} | Teste (15%): {X_test.shape[0]} amostras")
    print("-" * 50)
    
    resultados = {}

    # 5. PREPARAÇÃO DOS DADOS PARA OLS (Coerção e Constante)
    X_train_float = X_train.astype(np.float64)
    X_val_float = X_val.astype(np.float64)
    X_test_float = X_test.astype(np.float64)
    y_train_float = y_train.astype(np.float64)

    X_train_sm = sm.add_constant(X_train_float)
    X_val_sm = sm.add_constant(X_val_float)
    X_test_sm = sm.add_constant(X_test_float)

    print("### INICIANDO TREINAMENTO E VALIDAÇÃO ###")

    # ----------------------------------
    # 5.0 OLS REGRESSOR (BASELINE)
    # ----------------------------------
    ols_model = sm.OLS(y_train_float, X_train_sm).fit()
    r2_val_ols, rmse_val_ols, mae_val_ols = calcular_metricas(ols_model, X_val_sm, y_val)
    resultados['OLS'] = {'R2_val': r2_val_ols, 'model': ols_model, 'X_test': X_test_sm}
    print(f"OLS - R² Validação: {r2_val_ols:.4f}")

    # ----------------------------------
    # 5.1 GBR REGRESSOR
    # ----------------------------------
    gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, min_samples_split=5, min_samples_leaf=3, subsample=0.8, random_state=RANDOM_STATE)
    gbr.fit(X_train, y_train)
    r2_val_gbr, rmse_val_gbr, mae_val_gbr = calcular_metricas(gbr, X_val, y_val)
    resultados['GBR'] = {'R2_val': r2_val_gbr, 'model': gbr, 'X_test': X_test}
    print(f"GBR - R² Validação: {r2_val_gbr:.4f}")

    # ----------------------------------
    # 5.2 RANDOM FOREST REGRESSOR (RFR)
    # ----------------------------------
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=3, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    r2_val_rf, rmse_val_rf, mae_val_rf = calcular_metricas(rf, X_val, y_val)
    resultados['RFR'] = {'R2_val': r2_val_rf, 'model': rf, 'X_test': X_test}
    print(f"RFR - R² Validação: {r2_val_rf:.4f}")

    print("-" * 50)

    # ----------------------------------------------------------------------
    # 6. SELEÇÃO, AVALIAÇÃO FINAL E SERIALIZAÇÃO
    # ----------------------------------------------------------------------
    melhor_modelo_nome = max(resultados, key=lambda nome: resultados[nome]['R2_val'])
    melhor_resultado = resultados[melhor_modelo_nome]
    melhor_modelo = melhor_resultado['model']

    # Avaliação Final no Teste
    X_final_test = melhor_resultado['X_test']
    r2_final, rmse_final, mae_final = calcular_metricas(melhor_modelo, X_final_test, y_test)

    # Serialização
    joblib.dump(melhor_modelo, NOME_MODELO_SERIALIZADO)

    print(f"🏆 MODELO VENCEDOR (Baseado na Validação): {melhor_modelo_nome}")
    print(f"R² Final (Teste): {r2_final:.4f} | RMSE Final (Teste): {rmse_final:.4f}")
    print(f"✅ Modelo serializado em '{NOME_MODELO_SERIALIZADO}'")

# ----------------------------------------------------------------------
# LÓGICA DE EXECUÇÃO PARA REUSO DO MÓDULO (MLOps)
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # Este bloco simula o fluxo completo, consumindo o processamento
    
    # OBS: Para rodar este bloco, você precisará garantir que:
    # 1. Os módulos data_ingestion e data_processing estão acessíveis.
    # 2. O df_scaled e a lista de features foram gerados e salvos (ou carregados).
    
    # SIMULAÇÃO DE CARGA DO DF_SCALED E FEATURES (Substitua pela sua lógica de carga real)
    
    # Importar a função do processamento
    # from data_processing import processar_e_escalar_dados # Descomente se rodar em cadeia
    # df_long = carregar_e_combinar_dados()
    # df_scaled, features_finais = processar_e_escalar_dados(df_long)
    
    # ASSUMIR que df_scaled e features_finais estão carregados:
    # df_scaled = pd.read_pickle('df_scaled_temp.pkl') 
    # features_finais = ['scaled_Educacao_per_capita', ...] 
    
    # treinar_e_avaliar_modelos(df_scaled, features_finais)
    print("Módulo modeling.py configurado. Execute o pipeline principal para testar.")