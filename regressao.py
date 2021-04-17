import sys
import pandas as pd
import numpy as np


def compute_cost(thetas, x, y):
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

