# data_ingestion.py

import pandas as pd
from sqlalchemy import create_engine
import numpy as np

# 1. Configurar credenciais do banco (preencha com as suas informações)
db_config = {
    "host": "DB_HOST",
    "port": "DB_PORT",
    "user": "DB_USER",
    "password": "DB_PASSWORD",
    "dbname": "DB_NAME"
}

# 2. Criar engine de conexão (mantido fora da função para reuso)
engine = create_engine(
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
)

# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE INGESTÃO E COMBINAÇÃO
# ----------------------------------------------------------------------

def carregar_e_combinar_dados():
    """
    Carrega os dados FIMBRA, ENEM e tabelas auxiliares do banco de dados,
    realiza o merge sequencial e calcula a despesa per capita.
    Retorna o DataFrame final no formato Long.
    """
    
    # 3. Carregar a tabela FIMBRA 2023 inteira
    query_fimbra = "SELECT * FROM fimbra_2023;"
    df_fimbra = pd.read_sql(query_fimbra, engine)
    
    # 4. Carregar a tabela de candidatos ENEM
    query_enem = "SELECT * FROM candidatos_enem;"
    chunksize = 50000  # Carregar 50.000 linhas por vez (ajustável)
    chunks = []

    print(f"-> Carregando ENEM em partes (chunksize={chunksize})...")

    try:
        # O loop de leitura faz parte do bloco 'try'
        for chunk in pd.read_sql(query_enem, engine, chunksize=chunksize):
            chunks.append(chunk)
        # Opcional: Imprimir feedback para saber que está progredindo
        # print(f"   ... Carregadas {len(chunks) * chunksize} linhas")  # Manter a indentação

        # Se o loop terminar com sucesso, concatene o DataFrame
        df_enem = pd.concat(chunks, ignore_index=True)
        print(f"-> Carregamento do ENEM concluído. Total de linhas: {df_enem.shape[0]}")

    except Exception as e:
        # Se qualquer erro (timeout, etc.) ocorrer dentro do 'try', este bloco é executado
        print(f"ERRO CRÍTICO ao carregar ENEM em chunks ({e}). Tentando carregamento direto como fallback.")
    
        # Tenta o carregamento direto como fallback (se a falha foi por recursos, isso pode falhar)
        df_enem = pd.read_sql(query_enem, engine)
    
    # 5. Carregar tabela de municípios
    query_municipio = "SELECT * FROM municipio;"
    df_municipio = pd.read_sql(query_municipio, engine)
    
    # 6. Carregar tabela de população
    query_populacao = "SELECT * FROM populacao;"
    df_populacao = pd.read_sql(query_populacao, engine)
    
    # 7. Carregar tabela de unidades federativas (UF)
    query_uf = "SELECT * FROM unidade_federacao;"
    df_uf = pd.read_sql(query_uf, engine)

    # ------------------------------
    # 1. Filtrar candidatos ENEM 15-19 anos
    # ------------------------------
    df_enem_15_19 = df_enem[(df_enem['idade'] >= 15) & (df_enem['idade'] <= 19)].copy()

    # ------------------------------
    # 2. Agregar ENEM por município (mediana)
    # ------------------------------
    cols_notas = [
        'nota_ciencias_da_natureza', 'nota_ciencias_humanas',
        'nota_linguagens_e_codigos', 'nota_matematica',
        'nota_redacao', 'nota_media'
    ]
    df_enem_agg = df_enem_15_19.groupby('codigo_municipio_prova')[cols_notas].median().reset_index()

    # ------------------------------
    # 3. Merge ENEM agregado com MUNICIPIO
    # ------------------------------
    df_enem_mun = pd.merge(
        df_enem_agg,
        df_municipio[['codigo_municipio', 'codigo_municipio_dv', 'nome_municipio', 'cd_uf']],
        left_on='codigo_municipio_prova',
        right_on='codigo_municipio_dv',
        how='left'
    )

    # ------------------------------
    # 4. Merge com POPULAÇÃO
    # ------------------------------
    df_enem_pop = pd.merge(
        df_enem_mun,
        df_populacao[['codigo_municipio_dv', 'numero_habitantes', 'faixa_populacao']],
        left_on='codigo_municipio_prova',
        right_on='codigo_municipio_dv',
        how='left'
    )

    # ------------------------------
    # 5. Merge com UF
    # ------------------------------
    df_enem_pop_uf = pd.merge(
        df_enem_pop,
        df_uf[['cd_uf', 'sigla_uf']],
        left_on='cd_uf',
        right_on='cd_uf',
        how='left'
    )

    # ------------------------------
    # 6. Merge final com FIMBRA
    # ------------------------------
    df_combined_final = pd.merge(
        df_fimbra,
        df_enem_pop_uf,
        left_on='cod_ibge',
        right_on='codigo_municipio_dv_x',
        how='inner'
    )

    # 7. Limpeza e Feature Engineering (Long Format)
    # 1. Remover a coluna extra (codigo_municipio_dv_y)
    df = df_combined_final.drop(columns=['codigo_municipio_dv_y'])

    # 2. Criar coluna de despesa per capita
    df['despesa_per_capita'] = df['valor_despesa'] / df['numero_habitantes']
    
    # Retorna o DF Long (df) e o DF de Filtros (sem duplicatas de município/ENEM)
    df_enem_agg_for_filters = df_enem_pop_uf.drop_duplicates(subset=['codigo_municipio_prova']).reset_index(drop=True)

    return df, df_enem_agg_for_filters # Retorna o DF de Análise e o DF de Filtros

# ----------------------------------------------------------------------
# LÓGICA DE EXECUÇÃO PARA REUSO DO MÓDULO (MLOps)
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # Este bloco só será executado se o script for rodado diretamente
    df_raw = carregar_e_combinar_dados()

    print(f"Ingestão e combinação concluídas. DataFrame (Long Format) pronto para processamento. Shape: {df_raw.shape}")
