#하나의 텍스트 파일로 training set으로 설정하여 읽어온다

import tensorflow as tf
import numpy as np #파일을 한줄씩 읽어오기 위해 사용

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
# #표시는 읽지 않음
# 어떤것으로 행을 나눌지 unpack
# dtype 어느 타입으로 읽어올 것인지
x_data = xy[0:-1] # 0에서 전체-1
y_data = xy[-1] #끝에있는 데이터(-1)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0)) #1,x_data의 길이만큼 설정
#b = tf.Variable(tf.random_uniform([1], -1, 1))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)  # goal is minimize cost

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if(step % 20 == 0):
        print(step, sess.run(cost), sess.run(W))
