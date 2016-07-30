import tensorflow as tf

a=tf.placeholder(tf.int16) # setting parameter type
b=tf.placeholder(tf.int16)

add=tf.add(a,b)
mul=tf.mul(a,b)

with tf.Session() as sess:
    print("%i" % sess.run(add,feed_dict={a:2, b:3}))
    print("%i" % sess.run(mul, feed_dict={a: 2, b: 3}))