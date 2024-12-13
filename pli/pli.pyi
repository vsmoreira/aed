import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import pulp as pl
import time


# Gerador de diplomatas e postos baseado na nova lógica
def gerar_base_diplomatas(qtd_diplomatas, qtd_postos):
    ids = []
    pedagios = []
    lotacoes = []
    tempos_servico = []
    tempos_lotacao = []
    tempos_fora = []

    for p in range(qtd_diplomatas):
        if len(lotacoes) < qtd_postos:
            while True:
                lotacao = 'P' + str(np.random.randint(1, qtd_postos + 1))
                if lotacao == 'P0' or lotacao not in lotacoes:
                    break
        else:
            lotacao = 'P0'

        tempo_servico = np.random.randint(300)
        pedagio = 0
        tempo_na_lotacao = 0
        tempo_fora = 0
        if lotacao == 'P0':  # Ele está no Brasil
            pedagio = np.random.randint(30)
            tempo_na_lotacao = 0
            tempo_fora = 0
        else:
            pedagio = 0
            tempo_na_lotacao = np.random.randint(25)
            tempo_fora = tempo_na_lotacao + np.random.randint(97, 122)

        ids.append('D' + str(p + 1))
        pedagios.append(pedagio)
        lotacoes.append(lotacao)
        tempos_servico.append(tempo_servico)
        tempos_lotacao.append(tempo_na_lotacao)
        tempos_fora.append(tempo_fora)

    d = {
        'id': ids,
        'pedagio': pedagios,
        'lotacao': lotacoes,
        'tempo_servico': tempos_servico,
        'tempo_na_lotacao': tempos_lotacao,
        'tempo_fora': tempos_fora
    }
    return pd.DataFrame(data=d)


def gerar_base_postos(qtd_postos):
    postos = ['P0']
    classificacoes = ['*']
    tipos_classificacoes = ['A', 'B', 'C', 'D']

    for p in range(qtd_postos):
        posto = 'P' + str(p + 1)
        postos.append(posto)
        classificacoes.append(tipos_classificacoes[np.random.randint(4)])

    d = {
        'posto': postos,
        'classificacao': classificacoes
    }

    return pd.DataFrame(data=d)


# Função de custo compartilhada
def calcular_custo_aresta(diplomata, posto, cidades):
    CUSTO_INFINITO = 999999999.99
    if diplomata.tempo_fora >= 144:
        if posto.classificacao == '*':
            return 0.0
        else:
            return CUSTO_INFINITO

    if diplomata.tempo_na_lotacao >= 24 and diplomata.lotacao == posto.posto:
        return CUSTO_INFINITO

    if posto.classificacao == 'A':
        return 1.0
    if posto.classificacao == 'B':
        return 2.0
    if posto.classificacao == 'C':
        return 3.0
    if posto.classificacao == 'D':
        return 4.0

    return CUSTO_INFINITO


# Função para calcular pesos (custos)
def calcular_custos_arestas(diplomatas, cidades):
    pesos_arestas = []
    for _, diplomata in diplomatas.iterrows():
        pesos_diplomata = []
        for _, posto in cidades.iterrows():
            pesos_diplomata.append(calcular_custo_aresta(diplomata, posto, cidades))
        pesos_arestas.append(pesos_diplomata)
    return pesos_arestas


# Algoritmo Húngaro
def rodar_hungaro(diplomatas, postos):
    pesos_arestas = calcular_custos_arestas(diplomatas, postos)
    matriz_custos = np.array(pesos_arestas)
    linhas, colunas = linear_sum_assignment(matriz_custos)
    custo_total = matriz_custos[linhas, colunas].sum()
    tempo_execucao = time.time()
    return linhas, colunas, custo_total, tempo_execucao


# Algoritmo PLI
def rodar_pli(diplomatas, postos, num_diplomatas, num_postos):
    problema = pl.LpProblem("Alocacao_Diplomatas", pl.LpMinimize)
    variaveis = {}
    for i in range(num_diplomatas):
        for j in range(num_postos):
            variaveis[(i, j)] = pl.LpVariable(f"x_{i}_{j}", cat="Binary")

    # Objetivo
    problema += pl.lpSum(
        variaveis[(i, j)] * calcular_custo_aresta(diplomatas.iloc[i], postos.iloc[j], postos)
        for i in range(num_diplomatas)
        for j in range(num_postos)
    )

    # Restrições
    for i in range(num_diplomatas):
        problema += pl.lpSum(variaveis[(i, j)] for j in range(num_postos)) == 1

    for j in range(num_postos):
        problema += pl.lpSum(variaveis[(i, j)] for i in range(num_diplomatas)) <= 1

    problema.solve()

    alocacoes = [
        (i, j) for i in range(num_diplomatas) for j in range(num_postos)
        if pl.value(variaveis[(i, j)]) == 1
    ]
    custo_total = pl.value(problema.objective)
    tempo_execucao = time.time()

    return alocacoes, custo_total, tempo_execucao


# Teste comparativo
def comparar_algoritmos(qtd_diplomatas, qtd_postos):
    # Gerar dados
    diplomatas = gerar_base_diplomatas(qtd_diplomatas, qtd_postos)
    postos = gerar_base_postos(qtd_postos)

    # Rodar cada algoritmo
    inicio = time.time()
    linhas_h, colunas_h, custo_h, tempo_h = rodar_hungaro(diplomatas, postos)
    fim_h = time.time()
    tempo_h = fim_h - inicio

    inicio = time.time()
    alocacoes_p, custo_p, tempo_p = rodar_pli(diplomatas, postos, qtd_diplomatas, qtd_postos)
    fim_p = time.time()
    tempo_p = fim_p - inicio

    # Resultados finais
    df_resultados = pd.DataFrame({
        "Algoritmo": ["Húngaro", "PLI"],
        "Custo Total": [custo_h, custo_p],
        "Tempo de Execução (s)": [tempo_h, tempo_p]
    })

    return df_resultados


# Exemplo de uso
qtd_diplomatas = 350
qtd_postos = 227
resultados = comparar_algoritmos(qtd_diplomatas, qtd_postos)
print(resultados)


# Função para calcular médias por classificação
import pandas as pd

def calcular_classificacao_pli(diplomatas, postos, alocacoes):
    resultados = {
        "classificacao_anterior": [],
        "tempo_servico": [],
        "tempo_na_lotacao": [],
        "tempo_fora": [],
        "tempo_sede": [],
        "classificacao_alocada": []
    }

    for alocacao in alocacoes:
        id_diplomata, id_posto = alocacao

        diplomata = diplomatas.iloc[id_diplomata]
        posto = postos.iloc[id_posto]

        try:
            classificacao_anterior = diplomata["classificacao"] if "classificacao" in diplomata else "*"
            tempo_servico = float(diplomata["tempo_servico"])
            tempo_na_lotacao = float(diplomata["tempo_na_lotacao"])
            tempo_fora = float(diplomata["tempo_fora"])
            tempo_sede = float(diplomata["pedagio"])
            classificacao_alocada = posto["classificacao"]
        except (ValueError, KeyError, TypeError):
            continue

        resultados["classificacao_anterior"].append(classificacao_anterior)
        resultados["tempo_servico"].append(tempo_servico)
        resultados["tempo_na_lotacao"].append(tempo_na_lotacao)
        resultados["tempo_fora"].append(tempo_fora)
        resultados["tempo_sede"].append(tempo_sede)
        resultados["classificacao_alocada"].append(classificacao_alocada)

    df_resultados = pd.DataFrame(resultados)

    # Agrupando e calculando as médias, ignorando valores inválidos
    estatisticas_anteriores = df_resultados.groupby("classificacao_anterior").mean(numeric_only=True)
    estatisticas_alocadas = df_resultados.groupby("classificacao_alocada").mean(numeric_only=True)

    return estatisticas_anteriores, estatisticas_alocadas

# Exemplo de uso
# Supor que `alocacoes` foi obtido do PLI e contém pares (id_diplomata, id_posto)
diplomatas = gerar_base_diplomatas(qtd_diplomatas, qtd_postos)
postos = gerar_base_postos(qtd_postos)
alocacoes, custo_total, tempo_execucao = rodar_pli(diplomatas, postos, qtd_diplomatas, qtd_postos)

estatisticas_anteriores, estatisticas_alocadas = calcular_classificacao_pli(diplomatas, postos, alocacoes)

print("Estatísticas antes da realocação:")
print(estatisticas_anteriores)

print("Estatísticas após a realocação:")
print(estatisticas_alocadas)




import pandas as pd

def comparar_algoritmos(estatisticas_hungaro_anteriores, estatisticas_hungaro_alocadas, estatisticas_pli_anteriores, estatisticas_pli_alocadas):
    comparativo = {
        "Classificação": [],
        "Média Tempo Serviço (Húngaro Antes)": [],
        "Média Tempo Serviço (Húngaro Depois)": [],
        "Média Tempo Serviço (PLI Antes)": [],
        "Média Tempo Serviço (PLI Depois)": []
    }

    classificacoes = set(estatisticas_hungaro_anteriores.index).union(estatisticas_pli_anteriores.index)

    for classificacao in classificacoes:
        comparativo["Classificação"].append(classificacao)
        comparativo["Média Tempo Serviço (Húngaro Antes)"].append(
            estatisticas_hungaro_anteriores.get("tempo_servico", {}).get(classificacao, None)
        )
        comparativo["Média Tempo Serviço (Húngaro Depois)"].append(
            estatisticas_hungaro_alocadas.get("tempo_servico", {}).get(classificacao, None)
        )
        comparativo["Média Tempo Serviço (PLI Antes)"].append(
            estatisticas_pli_anteriores.get("tempo_servico", {}).get(classificacao, None)
        )
        comparativo["Média Tempo Serviço (PLI Depois)"].append(
            estatisticas_pli_alocadas.get("tempo_servico", {}).get(classificacao, None)
        )

    df_comparativo = pd.DataFrame(comparativo)
    return df_comparativo

# Ajuste na chamada da função rodar_hungaro
linhas_h, colunas_h, _, _ = rodar_hungaro(diplomatas, postos)
alocacoes_hungaro = list(zip(linhas_h, colunas_h))

estatisticas_hungaro_anteriores, estatisticas_hungaro_alocadas = calcular_classificacao_pli(diplomatas, postos, alocacoes_hungaro)

# Comparar os dois algoritmos
comparativo = comparar_algoritmos(estatisticas_hungaro_anteriores, estatisticas_hungaro_alocadas,
                                  estatisticas_anteriores, estatisticas_alocadas)

# Exibir o resultado completo
print("Comparação entre os Algoritmos Húngaro e PLI:")
print(comparativo.to_string(index=False))

distribuicao_antes = estatisticas_anteriores['tempo_servico'].count()
distribuicao_depois = estatisticas_alocadas['tempo_servico'].count()

print("Distribuição antes da realocação:")
print(distribuicao_antes)
print("Distribuição após a realocação:")
print(distribuicao_depois)


custo_medio_antes = estatisticas_anteriores['tempo_servico'].mean()
custo_medio_depois = estatisticas_alocadas['tempo_servico'].mean()

print("Custo médio por classificação antes:")
print(custo_medio_antes)
print("Custo médio por classificação após:")
print(custo_medio_depois)

desvio_antes = estatisticas_anteriores[['tempo_servico', 'tempo_na_lotacao', 'tempo_fora']].std()
desvio_depois = estatisticas_alocadas[['tempo_servico', 'tempo_na_lotacao', 'tempo_fora']].std()

print("Desvio padrão antes:")
print(desvio_antes)
print("Desvio padrão após:")
print(desvio_depois)

# Ordenar e redefinir os índices dos diplomatas e postos
diplomatas_sorted = diplomatas.sort_values(by='id').reset_index(drop=True)
postos_sorted = postos.sort_values(by='posto').reset_index(drop=True)

