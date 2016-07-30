import tensorflow as tf

x_data=[[1,1,1,1,1],
        [0., 2., 0., 4., 0.],
        [1., 0., 3., 0., 5.]] # 전체 데이터 부분에 1,1,1,1,1 이라는 부분이 들어가게 됨
y_data=[1,2,3,4,5]

W=tf.Variable(tf.random_uniform([1,3],-1.0,1.0)) # b를 추가하지 않아도 됨. 기존 x_data

hypothesis=tf.matmul(W,x_data)

cost=tf.reduce_mean(tf.square(hypothesis-y_data))

a=tf.Variable(0.1)
optimizer=tf.train.GradientDescentOptimizer(a)
train=optimizer.minimize(cost)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if(step%20==0):
        print(step, sess.run(cost), sess.run(W)) 
            #W에서 b를 포함하므로 따로 b를 출력해줄 필요가 없음

