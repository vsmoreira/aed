import pulp as pl
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import networkx as nx

from hungaro.hungaro import gerar_base_diplomatas

def carregar_postos(caminho_json):
    with open(caminho_json, "r", encoding="utf-8") as file:
        dados_postos = json.load(file)
    df_postos = pd.DataFrame(dados_postos)
    df_postos.rename(columns={
        "coUnidadeAdministrativa": "posto",
        "tpClassePosto": "classificacao",
        "coTipoPosto": "tipo"
    }, inplace=True)
    return df_postos

def get_posto_trabalho(id, postos):
    resultado = postos.loc[postos['posto'] == id].to_dict(orient='records')
    if resultado:
        return resultado[0]
    else:
        return None

# Valores constantes para maior clareza e manutenção
CUSTO_INFINITO = 999999999.99
TEMPO_MAXIMO_FORA = 144
TEMPO_MINIMO_PEDAGIO = 24
TEMPO_MINIMO_LOTACAO = 24


def calcular_custo_aresta(diplomata, posto, postos):
    def classificacao_para_custo(classificacao):
        custos = {'A': 0.0, 'B': 10.0, 'C': 100.0, 'D': 1000.0}
        return custos.get(classificacao, CUSTO_INFINITO)

    if diplomata.tempo_fora >= TEMPO_MAXIMO_FORA:
        return 0.0 if posto.classificacao == '*' else CUSTO_INFINITO

    if diplomata.tempo_na_lotacao >= TEMPO_MINIMO_LOTACAO and diplomata.lotacao == posto.posto:
        return CUSTO_INFINITO

    if diplomata.lotacao != 'P0':
        posto_trabalho_atual = get_posto_trabalho(diplomata.lotacao, postos)
        if (
                posto_trabalho_atual and
                diplomata.lotacao != posto.posto and
                posto.classificacao == 'A' and
                posto_trabalho_atual.get('classificacao') == 'A'
        ):
            return CUSTO_INFINITO
    elif diplomata.pedagio < TEMPO_MINIMO_PEDAGIO and posto.classificacao == 'A':
        return CUSTO_INFINITO

    return classificacao_para_custo(posto.classificacao)

def alocar_diplomatas_pli(diplomatas, postos):
    problema = pl.LpProblem("Alocacao_Diplomatas", pl.LpMinimize)
    variaveis = {}
    for i, diplomata in diplomatas.iterrows():
        for j, posto in postos.iterrows():
            variaveis[(i, j)] = pl.LpVariable(f"x_{i}_{j}", cat="Binary")

    problema += pl.lpSum(
        variaveis[(i, j)] * calcular_custo_aresta(diplomatas.iloc[i], postos.iloc[j], postos)
        for i in range(len(diplomatas)) for j in range(len(postos))
    )

    for i in range(len(diplomatas)):
        problema += pl.lpSum(variaveis[(i, j)] for j in range(len(postos))) == 1

    for j in range(len(postos)):
        problema += pl.lpSum(variaveis[(i, j)] for i in range(len(diplomatas))) <= 1

    problema.solve()

    alocacoes = [
        (i, j) for i in range(len(diplomatas)) for j in range(len(postos))
        if pl.value(variaveis[(i, j)]) == 1
    ]
    custo_total = pl.value(problema.objective)

    return alocacoes, custo_total

def executar_pli(caminho_json, num_diplomatas):
    df_postos = carregar_postos(caminho_json)
    diplomatas = gerar_base_diplomatas(num_diplomatas, len(df_postos))
    postos = df_postos

    inicio = time.time()
    alocacoes, custo_total = alocar_diplomatas_pli(diplomatas, postos)
    fim = time.time()

    tempo_execucao = fim - inicio

    resultados = pd.DataFrame([
        {"Diplomata": diplomatas.iloc[i].id, "Posto": postos.iloc[j].posto}
        for i, j in alocacoes
    ])

    return resultados, custo_total, tempo_execucao, diplomatas, postos, alocacoes


# 2. Construir matriz de custos - Mapa de Calor
# matriz_custos = np.zeros((len(diplomatas), len(postos)))
# for i in range(len(diplomatas)):
#     for j in range(len(postos)):
#         matriz_custos[i, j] = calcular_custo_aresta(diplomatas.iloc[i], postos.iloc[j], postos)

# 3. Plotar heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(matriz_custos, cmap="YlGnBu", annot=False)
# plt.title("Mapa de Calor dos Custos de Alocação")
# plt.xlabel("Postos")
# plt.ylabel("Diplomatas")
# plt.show()

# 3. Gráfico de Barras
# diplomata_ids = [diplomatas.iloc[i].id for i, j in alocacoes]
#
# plt.figure(figsize=(12, 6))
# plt.bar(diplomata_ids, alocacao_custos)
# plt.title("Custo de Alocação por Diplomata")
# plt.xlabel("Diplomata ID")
# plt.ylabel("Custo")
# plt.show()

# 5. Criar grafo - Gráfico de Rede
def criar_grafo_rede(diplomatas, postos, alocacoes):
    G = nx.DiGraph()
    for i, j in alocacoes:
        diplomata = diplomatas.iloc[i]
        posto = postos.iloc[j]
        custo = calcular_custo_aresta(diplomata, posto, postos)
        G.add_edge(f"Diplomata {diplomata.id}", f"Posto {posto.posto}", weight=custo)

    # Desenhar grafo
    pos = nx.spring_layout(G)  # Experimente circular_layout ou kamada_kawai_layout
    plt.figure(figsize=(16, 16))  # Aumentar tamanho da figura para caber mais informações
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=300,         # Tamanho reduzido dos nós
        font_size=4,           # Fonte menor
        width=0.5,             # Linhas mais finas
        edge_color='gray',     # Cor cinza para as linhas
        alpha=0.7              # Transparência para reduzir poluição visual
    )
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)  # Fonte menor para pesos
    plt.title("Grafo de Alocação", fontsize=12)
    plt.show()


    # Exemplo de uso
caminho_json = "/Users/andersoncaxeta/IdeaProjects/AEDI/db/postos.json"
num_diplomatas = 350

resultados, custo_total, tempo_execucao, diplomatas, postos, alocacoes = executar_pli(caminho_json, num_diplomatas)

# Mostrar resultados
print("Alocações (diplomata -> posto):")
print(resultados)
print("Custo total:", custo_total)
print("Tempo de execução:", tempo_execucao)

# Criar e mostrar o gráfico de rede
criar_grafo_rede(diplomatas, postos, alocacoes)

# 6.Interface Gráfica com Streamlit
# st.title("Resultados da Alocação de Diplomatas")
# st.write(resultados)
#
# # Adicionar gráfico interativo
# st.bar_chart(alocacao_custos)