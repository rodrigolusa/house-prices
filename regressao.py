import sys
import pandas as pd
import numpy as np


def compute_cost(thetas, x, y):
    """
    Calcula o erro quadratico medio

    Args:
        thetas (np.array): pesos dos atributos (1, m)
        x (np.array): variaveis independentes da regressao (n, m)
        y (np.array): variaveis dependentes da regressao  (1, m)

    Retorna:
        float: o erro quadratico medio
    """
    sum_ = np.sum(np.square((np.sum(np.dot(thetas, x.T))) - y))
    total_cost = sum_ / y.shape[0]
    return total_cost


def get_min_max(x):
    """ normalizacao min_max do vetor

    Args:
        Args:
        x (np.array): array
    Retorna:
        np.array: array normalizado
    """
    minimo = x.min()
    maximo = x.max()
    amplitude = maximo - minimo
    x = (x - minimo) / amplitude
    return x


def step_gradient(thetas_current, x, y, alpha):
    """Calcula um passo em direção ao EQM mínimo

    Args:
        thetas (np.array): pesos dos atributos (1, m)
        x (np.array): variaveis independentes da regressao (n, m)
        y (np.array): variaveis independente da regressao  (1, m)
    Retorna:
        np.array:  os novos valores de theta
    """
    derivada_funcao_erro = (np.dot(thetas_current, x.T) - y).reshape(1, x.shape[0])
    gradientes = (2 / x.shape[0]) * np.dot(derivada_funcao_erro, x)
    thetas_updated = thetas_current - (alpha * gradientes)
    return thetas_updated


def gradient_descent(x, y, starting_thetas=None, learning_rate=0.000001, num_iterations=10):
    """executa a descida do gradiente

    Args:
        x (np.array): variaveis independentes da regressao (n, m)
        y (np.array): variaveis independente da regressao  (1, m)
        starting_theta (np.array): pesos dos atributos (m, 1)
        learning_rate (float): hyperparâmetro para ajustar o tamanho do passo durante a descida do gradiente
        num_iterations (int): hyperparâmetro que decide o número de iterações que cada descida de gradiente irá executar

    Retorna:
        list : o primeiros parâmetro eh um np.array dos melhores thetas
               O segundo eh um np.array o historico de custo
               O terceiro eh uma matriz com o histórico dos thetas
    """
    # valores iniciais
    if starting_thetas:
        thetas = np.array(starting_thetas).reshape(1, x.shape[1])
    else:
        thetas = np.zeros(x.shape[1]).reshape(1, x.shape[1])

    cost_graph = []

    thetas_progress = []

    num_iterations = num_iterations
    for i in range(num_iterations):
        cost = compute_cost(thetas, x, y)
        cost_graph.append(cost)
        thetas = step_gradient(thetas, x, y, alpha=learning_rate)
        # print(thetas)
        thetas_progress.append(thetas)

    return thetas, cost_graph, np.array(thetas_progress).T


def get_data(file, atributos):
    """faz a leitura do csv e transformacoes necessarias nos dados

    Args:
        file (string): caminho do arquivo csv onde estao os dados
        atributos (list string): lista das variaveis indepedentes utilizadas
    Retorna:
        tupla (np.array): matriz da variaveis independentes e uma lista das variaveis indepedentes.
    """
    data = pd.read_csv(file)[atributos]
    x = data.iloc[:, :-1].astype("float").values
    x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
    y = data.iloc[:, -1].astype("float").values

    for i in range(x.shape[1] - 1):
        x[:, i] = get_min_max(x[:, i])
    y = get_min_max(y)
    return x, y


atributos = [["GrLivArea", "SalePrice"],
             ["GrLivArea", "OverallQual", "SalePrice"],
             ["GrLivArea", "OverallQual", "OverallCond", "GarageArea", "YearBuilt", "SalePrice"]]

if __name__ == "__main__":
    nome_arquivo = sys.argv[1]
    epocas = int(sys.argv[2])
    opcao_atributo = int(sys.argv[3])
    atributo = atributos[opcao_atributo]
    X, Y = get_data(nome_arquivo, atributo)
    #starting_thetas = (np.random.randn(X.shape[1]) * np.sqrt(2/X.shape[1])).tolist()
    starting_thetas = None
    thetas, cost_graph, thetas_progres = gradient_descent(X, Y, starting_thetas=starting_thetas,
                                                          learning_rate=0.000001,
                                                          num_iterations=epocas)

    # #Imprimir parâmetros otimizados
    for index, theta in enumerate(thetas[0]):
        print(f'theta_{index}: {theta}')

    # #Imprimir erro com os parâmetros otimizados
    print(f'Erro quadratico medio: {compute_cost(thetas, X, Y)}')

    # print(cost_graph)
    #
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.plot(cost_graph)
    # plt.xlabel('No. de interações')
    # plt.ylabel('Custo')
    # plt.title('Custo por iteração')
    # plt.show()

