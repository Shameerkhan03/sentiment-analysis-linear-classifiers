import project1 as p1
import utils
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Problem 5
#-------------------------------------------------------------------------------

toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

T = 100
L = 0.2

thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)

def plot_toy_results(algo_name, thetas):
    print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
    print('theta_0 for', algo_name, 'is', str(thetas[1]))
    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

plot_toy_results('Perceptron', thetas_perceptron)
plot_toy_results('Average Perceptron', thetas_avg_perceptron)
plot_toy_results('Pegasos', thetas_pegasos)

def plot_theta_convergence(theta_history, title):
    diffs = np.linalg.norm(np.diff(theta_history, axis=0), axis=1)

    plt.figure()
    plt.plot(diffs)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("||Δθ||")
    plt.show()

plot_theta_convergence(thetas_perceptron[2], "Perceptron Convergence")
plot_theta_convergence(thetas_avg_perceptron[2], "Average Perceptron Convergence")
plot_theta_convergence(thetas_pegasos[2], "Pegasos Convergence")

