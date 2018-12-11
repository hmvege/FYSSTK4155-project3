import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tqdm import tqdm, trange

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


def lecture_notes_example():
    """Example taken from lecture notes"""
    Nx = 10
    x_np = np.linspace(0, 1, Nx)

    Nt = 10
    t_np = np.linspace(0, 1, Nt)

    X, T = np.meshgrid(x_np, t_np)

    x = X.ravel()
    t = T.ravel()

    # Construction of NN
    zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)), shape=(-1, 1))
    x = tf.reshape(tf.convert_to_tensor(x), shape=(-1, 1))
    t = tf.reshape(tf.convert_to_tensor(t), shape=(-1, 1))

    points = tf.concat([x, t], 1)

    num_iter = 100000
    num_hidden_neurons = [90]

    X = tf.convert_to_tensor(X)
    T = tf.convert_to_tensor(T)

    with tf.variable_scope("dnn"):
        num_hidden_layers = np.size(num_hidden_neurons)

        previous_layer = points

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(
                previous_layer, num_hidden_neurons[l], activation=tf.nn.sigmoid)
            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, 1)

    def u(x_):
        return tf.sin(np.pi*x_)

    def v(x_):
        return -np.pi*tf.sin(np.pi*x_)

    with tf.name_scope("loss"):
        g_trial = (1-t**2)*u(x)*tf.cos(np.pi*t) - \
            tf.sin(np.pi*x)*tf.sin(np.pi*t)*dnn_output

        g_trial_d2t = tf.gradients(tf.gradients(g_trial, t), t)
        g_trial_d2x = tf.gradients(tf.gradients(g_trial, x), x)

        loss = tf.losses.mean_squared_error(
            zeros, g_trial_d2t[0] - g_trial_d2x[0])

    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    g_analytic = tf.sin(np.pi*x)*tf.cos(np.pi*t) - \
        tf.sin(np.pi*x)*tf.sin(np.pi*t)
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

    G_analytic = g_analytic.reshape((Nt, Nx))
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
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Position $x$")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title("Analytical solution")
    s = ax.plot_surface(XX, TT, G_analytic, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Position $x$")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title("Difference")
    s = ax.plot_surface(XX, TT, diff, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Position $x$")
    plt.show()

if __name__ == '__main__':
    lecture_notes_example()
