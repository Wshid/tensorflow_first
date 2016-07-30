import tensorflow as tf
import numpy as np

xy=np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data=xy[0:-1] # 0부터 n-1 전까지 가져옴
y_data=xy[-1] # 마지막 번째 열만 가져옴

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W=tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, -1.0))

h=tf.matmul(W,X)
hypothesis=tf.div(1., 1. + tf.exp(-h)) #H(x)=1/(1+e^((-W)^T*X))

cost=-tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

a=tf.Variable(0.1)
optimizer=tf.train.GradientDescentOptimizer(a)
train=optimizer.minimize(cost)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if(step % 20 ==0):
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}),sess.run(W))