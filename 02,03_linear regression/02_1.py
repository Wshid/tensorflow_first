import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for w and b that compute y_data = W * x_data + b
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) 
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
## variable로 지정해야 나중에 업데이트가 가능
## -1에서 1까지의 랜덤하게 지정

# my hypothesis
hypothesis = w * x_data + b #linear 수식을 명명

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data)) ## cost수식 적용
##square : 제곱
##reduce_mean : 평균
# 물론 이런다고 해서 계산이 나는것이 아니라 연산 지정

# minimize
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a) ## 이렇게 해서 minimize하는 알고리즘 사용한다
train = optimizer.minimize(cost)

# before starting, initialize the variables
init = tf.initialize_all_variables() # 변수 선언 후, 초기화를 하지 않으면 에러 발생

# launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001): ##xrange의 경우 python3에서 명칭이 바뀜, range로 사용해야함
    sess.run(train)
    if(step % 20 == 0):
        print(step, sess.run(cost), sess.run(w), sess.run(b)) ## 이 for문을 가지고 실행

# learns best fit is w: [1] b: [0]
