#Multivariable Example
#H(x1,x2)= w1*x1+w2*x2+b

import tensorflow as tf

x1_data=[1,0,3,0,5]
x2_data=[0,2,0,4,0]
y_data=[1,2,3,4,5]

W1=tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # W는 범위 설정
W2=tf.Variable(tf.random_uniform([1],-1.0,1.0)) # W2역시 범위 설정

b= tf.Variable(tf.random_uniform([1],-1.0,1.0)) # b의 범위 역시 설정

hypothesis=W1*x1_data+W2*x2_data+b #multivariable에 맞게 가설을 세운 뒤

cost=tf.reduce_mean(tf.square(hypothesis-y_data)) # 최소제곱법을 이용하여 cost 계산

a=tf.Variable(0.1) # a의 값을 준후(아마 기울기 변동)
optimizer=tf.train.GradientDescentOptimizer(a) # 최적화 시작
train=optimizer.minimize(cost) #cost에 대해 최소화 하도록

init=tf.initialize_all_variables() # 모든 변수 초기화

sess=tf.Session() # 세션 열고
sess.run(init) # 연산 시작 - init

for step in range(2001):
    sess.run(train) # cost 최소화 연산 시작
    if(step % 20==0):
        print(step, sess.run(cost),sess.run(W1),sess.run(W2),sess.run(b))