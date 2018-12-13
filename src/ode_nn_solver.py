import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error

from tqdm import tqdm, trange

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


def tf_core(X, T, num_hidden_neurons, hidden_activation_function,
            optimizer, dropout_rate=0.0):

    x = X.ravel()
    t = T.ravel()

    # Construction of NN
    zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)), shape=(-1, 1))
    x = tf.reshape(tf.convert_to_tensor(x), shape=(-1, 1))
    t = tf.reshape(tf.convert_to_tensor(t), shape=(-1, 1))

    points = tf.concat([x, t], 1)

    X = tf.convert_to_tensor(X)
    T = tf.convert_to_tensor(T)

    # Sets up the layers
    with tf.variable_scope("dnn"):
        num_hidden_layers = np.size(num_hidden_neurons)

        previous_layer = points

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(
                previous_layer, num_hidden_neurons[l],
                activation=hidden_activation_function)

            # current_layer = tf.layers.dense(
            #     previous_layer, num_hidden_neurons[l],
            #     activation=tf.nn.sigmoid)

            if dropout_rate != 0.0:
                current_layer = tf.nn.dropout(current_layer, dropout_rate)

            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, 1)

    def u(x_):
        """Initial condition for t=0."""
        return tf.sin(np.pi*x_)  # Divide by L?

    def v(x_):
        """ du/dx """
        return -np.pi*tf.sin(np.pi*x_)

    # Sets up loss function
    with tf.name_scope("loss"):
        # Trial function
        h1 = (1 - t)*u(x)
        h2 = x*(1-x)*t*dnn_output
        g_trial = h1 + h2

        g_trial_dt = tf.gradients(g_trial, t)
        g_trial_d2x = tf.gradients(tf.gradients(g_trial, x), x)

        # Sets up loss function
        loss = tf.losses.mean_squared_error(
            zeros, g_trial_dt[0] - g_trial_d2x[0])

    # Sets up optimizer
    with tf.name_scope("train"):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    # Analytical solution, can keep here
    g_analytic = tf.sin(np.pi*x)*tf.exp(-np.pi*np.pi*t)
    g_dnn = None

    # Execution phase
    with tf.Session() as sess:
        init.run()
        for i in trange(num_iter, desc="Training dnn"):
            sess.run(training_op)

            if i % 100 == 0:
                tqdm.write("Cost: {0:.8f}".format(loss.eval()))

        g_analytic = g_analytic.eval()
        g_dnn = g_trial.eval()

    # Compare nn solution with analytical solution
    difference = np.abs(g_analytic - g_dnn)
    max_diff = np.max(difference)

    G_analytic = g_analytic.reshape((Nt, Nx))  # TODO: Need to reshape here?
    G_dnn = g_dnn.reshape((Nt, Nx))

    diff = np.abs(G_analytic - G_dnn)

    r2 = r2_score(g_analytic, g_dnn)
    mse = mean_squared_error(g_analytic, g_dnn)

    return G_analytic, G_dnn, diff, max_diff, r2, mse


def run():

    optimizers = [
        tf.train.AdamOptimizer(),
        tf.train.GradientDescentOptimizer(learning_rate)]

    activation_functions = [
        tf.nn.sigmoid,
        tf.nn.relu,
        tf.sigmoid,
        tf.tanh,
        tf.leaky_relu,
    ]

    


def task_c():
    x0 = 0.0
    L = 1.0
    t0 = 0.0
    t1 = 0.5
    t2 = 1.0

    Nx = 10
    Nt = 10

    num_iter = 100000
    num_hidden_neurons = [90]

    x_np = np.linspace(x0, L, Nx)
    t_np = np.linspace(t0, t1, Nt)

    X, T = np.meshgrid(x_np, t_np)

    x = X.ravel()
    t = T.ravel()

    # Construction of NN
    zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)), shape=(-1, 1))
    x = tf.reshape(tf.convert_to_tensor(x), shape=(-1, 1))
    t = tf.reshape(tf.convert_to_tensor(t), shape=(-1, 1))

    points = tf.concat([x, t], 1)

    X = tf.convert_to_tensor(X)
    T = tf.convert_to_tensor(T)

    # Sets up the layers
    with tf.variable_scope("dnn"):
        num_hidden_layers = np.size(num_hidden_neurons)

        previous_layer = points

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(
                previous_layer, num_hidden_neurons[l],
                activation=tf.nn.sigmoid)
            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, 1)

    def u(x_):
        """Initial condition for t=0."""
        return tf.sin(np.pi*x_)  # Divide by L?

    def v(x_):
        """ du/dx """
        return -np.pi*tf.sin(np.pi*x_)

    # Sets up loss function
    with tf.name_scope("loss"):
        # Trial function
        h1 = (1 - t)*u(x)
        h2 = x*(1-x)*t*dnn_output
        g_trial = h1 + h2

        g_trial_dt = tf.gradients(g_trial, t)
        g_trial_d2x = tf.gradients(tf.gradients(g_trial, x), x)

        # Sets up loss function
        loss = tf.losses.mean_squared_error(
            zeros, g_trial_dt[0] - g_trial_d2x[0])

    # Sets up optimizer
    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    # Analytical solution, can keep here
    g_analytic = tf.sin(np.pi*x)*tf.exp(-np.pi*np.pi*t)
    g_dnn = None

    # Execution phase
    with tf.Session() as sess:
        init.run()
        for i in trange(num_iter, desc="Training dnn"):
            sess.run(training_op)

            if i % 100 == 0:
                tqdm.write("Cost: {0:.8f}".format(loss.eval()))

        g_analytic = g_analytic.eval()
        g_dnn = g_trial.eval()

    # Compare nn solution with analytical solution
    difference = np.abs(g_analytic - g_dnn)
    print("Max absolute difference: ", np.max(difference))

    r2 = r2_score(g_analytic, g_dnn)
    print("R2 score: ", r2)

    G_analytic = g_analytic.reshape((Nt, Nx))  # TODO: Need to reshape here?
    G_dnn = g_dnn.reshape((Nt, Nx))

    diff = np.abs(G_analytic - G_dnn)

    # Plots results
    XX, TT = np.meshgrid(x_np, t_np)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title("Solution from the deep neural network w/ %d layer" %
                 len(num_hidden_neurons))
    s = ax.plot_surface(XX, TT, G_dnn, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Position $x$")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title("Analytical solution")
    s = ax.plot_surface(XX, TT, G_analytic, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Position $x$")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title("Difference")
    s = ax.plot_surface(XX, TT, diff, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Position $x$")
    plt.show()


def task_d():
    pass


def main():
    tf.reset_default_graph()
    task_c()
    # task_d()


if __name__ == '__main__':
    main()
