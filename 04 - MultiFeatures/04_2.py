import tensorflow as tf
x_data=[[0., 2., 0., 4., 0.], [1., 0., 3., 0., 5.]] #x_data역시 배열로 선언
y_data=[1,2,3,4,5]

W=tf.Variable(tf.random_uniform([1,2], -1.0, 1.0)) #배열의 크기가 증가하였음
b=tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #b의 경우 차이가 없음

hypothesis=tf.matmul(W, x_data)+b # Multi-variable에서 가장 핵심적인 부분
cost=tf.reduce_mean(tf.square(hypothesis-y_data))

a=tf.Variable(0.1) #W:=W-α*(1/m) * sum{(Wx-y)*x} 에서 α를 생각하면 될 듯
optimizer=tf.train.GradientDescentOptimizer(a)
train=optimizer.minimize(cost)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if(step%20==0):
        print(step, sess.run(cost), sess.run(W), sess.run(b))