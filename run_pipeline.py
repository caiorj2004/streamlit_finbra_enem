# run_pipeline.py

import os
from data_ingestion import carregar_e_combinar_dados
from data_processing import processar_e_escalar_dados
from modeling import treinar_e_avaliar_modelos

# --- VARIÁVEIS GLOBAIS ---
MODELS_DIR = 'models' 

def executar_pipeline_completo():
    """Executa o pipeline de MLOps: Ingestão -> Processamento -> Modelagem."""
    
    print("=============================================")
    print("=== INICIANDO PIPELINE MLOPS: FIMBRA x ENEM ===")
    print("=============================================")

    # 1. INGESTÃO: Carrega os dados brutos, faz o merge e retorna 2 DFs (df_long e df_filters)
    # MODIFICAÇÃO AQUI: CAPTURAR O RETORNO DUPLO
    df_long, df_filters = carregar_e_combinar_dados()
    print("-> 1. Ingestão e Merge Concluídos.")
    
    # --- PREPARAÇÃO DO AMBIENTE ---
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"-> Diretório '{MODELS_DIR}' criado.")

    # 2. PROCESSAMENTO: Salva DFs de EDA, Pivotagem, VIF, Padronização e Serialização
    # MODIFICAÇÃO AQUI: PASSAR AMBOS OS DFs
    df_scaled, features_finais = processar_e_escalar_dados(df_long, df_filters)
    print("-> 2. Processamento, VIF e Padronização Concluídos. Preprocessor e DFs de EDA salvos.")

    # 3. MODELAGEM: Treinamento, Seleção do Vencedor (RFR) e Serialização do Modelo
    treinar_e_avaliar_modelos(df_scaled, features_finais)
    print("-> 3. Modelagem e Serialização do Modelo Vencedor (RFR) Concluídas.")
    
    print("\n=============================================")
    print("=== PIPELINE EXECUTADO COM SUCESSO! ===========")
    print("=============================================")


if __name__ == '__main__':
    executar_pipeline_completo()