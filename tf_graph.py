import tensorflow as tf


graph1 = tf.Graph()
#
with graph1.as_default():
    b = tf.Variable(0.7)
    c = tf.Variable(0.9)
    a = b+c
    print(a)
    init = tf.global_variables_initializer()

graph2 = tf.Graph()
#
with graph2.as_default():
    b2 = tf.Variable(0.17)
    c2 = tf.Variable(0.19)
    a2 = b2 + c2
    print(a)
    init2 = tf.global_variables_initializer()
    with tf.Session() as sess2:
        sess2.run(init2)
        print(sess2.run(a2))
with tf.Session(graph=graph1) as sess2:
    sess2.run(init)
    print(sess2.run(a))
