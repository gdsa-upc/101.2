# -*- coding: utf-8 -*-

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow como tf

# Marcador de posición
x = tf.placeholder ("flotar", [Ninguno, 784])

# Variables iniciales, tensores llenos de ceros.
W = tf.Variable (tf.zeros ([784,10]))
b = tf.Variable (tf.zeros ([10]))

# Aplicamos el modelo. Multiplicamos x por W con la expresión, añadimos b, y finalmente aplicamos tf.nn.softmax.
y = tf.nn.softmax (tf.matmul (x, W) + b)


# FORMACIÓN:

# Marcador de posición para introducir las respuestas correctas:
y_ = tf.placeholder ("flotar", [Ninguno, 10])

# Cruz-entropía
cross_entropy = -tf.reduce_sum (y_ * tf.log (y))

# Algoritmo de optimización para modificar las variables y reducir el coste.
train_step = tf.train.GradientDescentOptimizer (0,01) .minimize (cross_entropy)

# Inicializar las variables que hemos creado.
init = tf.initialize_all_variables ()

# Lanzar el modelo en una sesión, y ejecutar la operación que inicializa las variables:
sess = tf.Session ()
sess.run (init)

for i in range (1000):
  batch_xs, batch_ys = mnist.train.next_batch (100)
  sess.run (train_step, feed_dict = {x: batch_xs, Y_: batch_ys})
  
  
# EVALUACIÓN DEL MODELO

# Para comprobar si nuestra predicción coincide con la verdad.
correct_prediction = tf.equal (tf.argmax (y, 1), tf.argmax (y_, 1))

# Para determinar qué fracción son correctos, echamos a los números de punto flotante y luego tomamos la media.
precisión = tf.reduce_mean (tf.cast (correct_prediction, "float"))

# Pedimos nuestra precisión en nuestros datos de prueba.
sess.run impresión (exactitud, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})