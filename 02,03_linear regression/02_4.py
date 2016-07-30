import tensorflow as tf

x_data=[1., 2., 3.]
y_data=[1., 2., 3.]

W=tf.Variable(tf.random_uniform([1],-10.0,10.0))

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

hypothesis=W*X #가설세우기

cost=tf.reduce_mean(tf.square(hypothesis-Y)) #최소제곱법

descent=W-tf.mul(0.1,tf.reduce_mean(tf.mul((tf.mul(W,X)-Y),X))) # α=0.1
update=W.assign(descent) #업데이트!

init=tf.initialize_all_variables() #세션을 통해 직접 실행해주어야 적용됨

sess=tf.Session()
sess.run(init)

for step in range(50):
    sess.run(update, feed_dict={X:x_data, Y:y_data}) # placeholder된 값을 출력
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
    # W값이 cost 함수의 미분값이므로 차차 1로 수렴하는 것을 볼 수 있다
    # Cost의 값이 0으로 수렴하는 것을 볼 수 있다
    # Gradient Descent Algorithm 함수의 원리를 확인할 수 있음
