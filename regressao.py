import sys
import pandas as pd
import numpy as np


def compute_cost(thetas, x, y):
    """
    Calcula o erro quadratico medio

    Args:
        theta_0 (float): intercepto da reta
        theta_1 (float): inclinacao da reta
        data (np.array): matriz com o conjunto de dados, x na coluna 0 e y na coluna 1

    Retorna:
        float: o erro quadratico medio
    """
    total_cost = 0
    sum_ = np.sum(np.square((thetas[0] + np.sum(np.dot(thetas[1:].T, x.T))) - y))

    total_cost = sum_ / y.shape[0]

    return total_cost


def get_min_max(x):
    minimo = x.min()
    maximo = x.max()
    amplitude = maximo - minimo
    x = (x - minimo) / amplitude
    return x


def get_derivada(x, y, thetas, derivada_theta):
    derivada_funcao_erro = thetas[0] + np.dot(thetas[1:].T, x.T) - y
    return (2 / x.shape[0]) * np.sum(derivada_funcao_erro * derivada_theta)


def step_gradient(thetas_current, x, y, alpha):
    """Calcula um passo em direção ao EQM mínimo

    Args:
        theta_0_current (float): valor atual de theta_0
        theta_1_current (float): valor atual de theta_1
        data (np.array): vetor com dados de treinamento (x,y)
        alpha (float): taxa de aprendizado / tamanho do passo

    Retorna:
        tupla: (theta_0, theta_1) os novos valores de theta_0, theta_1
    """
    thetas_updated = []
    for indice in range(0, x.shape[1]+1):
        if indice == 0:
            derivada_theta = 1
        else:
            derivada_theta = x[:, indice-1]
        derivada = get_derivada(x, y, thetas_current, derivada_theta)
        theta_updated = thetas_current[indice] - (alpha * derivada)
        thetas_updated.append(theta_updated)
    return np.array(thetas_updated).reshape(len(thetas_updated), 1)


def gradient_descent(x, y, starting_thetas=None, learning_rate=0.000001, num_iterations=10):
    """executa a descida do gradiente

    Args:
        data (np.array): dados de treinamento, x na coluna 0 e y na coluna 1
        starting_theta_0 (float): valor inicial de theta0
        starting_theta_1 (float): valor inicial de theta1
        learning_rate (float): hyperparâmetro para ajustar o tamanho do passo durante a descida do gradiente
        num_iterations (int): hyperparâmetro que decide o número de iterações que cada descida de gradiente irá executar

    Retorna:
        list : os primeiros dois parâmetros são o Theta0 e Theta1, que armazena o melhor ajuste da curva. O terceiro e quarto parâmetro, são vetores com o histórico dos valores para Theta0 e Theta1.
    """

    # valores iniciais
    if starting_thetas:
        thetas = np.array(starting_thetas).reshape(x.shape[1] + 1, 1)
    else:
        thetas = np.zeros(x.shape[1] + 1).reshape(x.shape[1] + 1, 1)

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []

    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    thetas_progress = []

    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    num_iterations = num_iterations
    for i in range(num_iterations):
        cost = compute_cost(thetas, x, y)
        cost_graph.append(cost)
        thetas = step_gradient(thetas, x, y, alpha=learning_rate)
        # print(thetas)
        thetas_progress.append(thetas)

    return thetas, cost_graph, np.array(thetas_progress).T


def get_data(file, atributos):
    data = pd.read_csv(file)[atributos]
    x = data.iloc[:, :-1].astype("float").values
    y = data.iloc[:, -1].astype("float").values

    for i in range(data.shape[1] - 1):
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
    #starting_thetas = (np.random.randn(X.shape[1]+1) * np.sqrt(2/X.shape[1]+1)).tolist()
    starting_thetas = None
    thetas, cost_graph, thetas_progres = gradient_descent(X, Y, starting_thetas=starting_thetas,
                                                          learning_rate=0.000002,
                                                          num_iterations=epocas)

    # #Imprimir parâmetros otimizados
    for index, theta in enumerate(thetas):
        print(f'theta_{index}: {theta[0]}')

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

