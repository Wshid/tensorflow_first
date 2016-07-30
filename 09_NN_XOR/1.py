#XOR with logistic regression
# 정확도가 너무 낮음, accuracy가 1이 되어도 문제였음

import tensorflow as tf
import numpy as np

xy=np.loadtxt('train.txt', unpack=True)
#x_data=xy[0:-1]
#y_data=xy[-1]
x_data=np.transpose(xy[0:-1])
y_data=np.reshape(xy[-1],(4,1))

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W=tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0)) #학습할 Weight 지정

h=tf.matmul(W,X)
hypothesis=tf.div(1., 1.+tf.exp(-h)) #sigmoid 함수
cost=-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis)) #cross-entropy

a=tf.Variable(0.01)
optimizer=tf.train.GradientDescentOptimizer(a)
train=optimizer.minimize(cost)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(1000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if(step%200==0):
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
            
    correct_prediction=tf.equal(tf.floor(hypothesis+0.5),Y) #Test Model, 0.5보다 클경우 1, floor는 1또는 0 반환, Y는 실제 값
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float")) #Calculate Accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
    print("Accuracy:", accuracy.eval({X:x_data, Y:y_data}))
    