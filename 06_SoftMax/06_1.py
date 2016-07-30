import tensorflow as tf
import numpy as np

xy=np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data=np.transpose(xy[0:3])
y_data=np.transpose(xy[3:])

X=tf.placeholder("float", [None, 3]) #x의 개수가 정해지지 않았으므로 None
Y=tf.placeholder("float", [None, 3])
W=tf.Variable(tf.zeros([3,3])) #W 벡터의 크기도 3 by 3으로 초기화

hypothesis=tf.nn.softmax(tf.matmul(X,W)) #곱할때 a*b b*c여야 하므로 위에서 transpose 진행

learning_rate=0.01

cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))
#cost fuction의 경우 D(S,L)-Cross Entropy를 최소제곱법으로 계산하면 됨
#reduction_indices=https://www.tensorflow.org/versions/r0.9/api_docs/python/math_ops.html#reduce_sum
#mean의 경우 최소제곱법이라 해서 sum에 1/n을 곱한 값
#sum의 경우 단순히 시그마 합을 의미

optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 200==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
    
    a=sess.run(hypothesis, feed_dict={X:[[1,11,7]]}) #[1 0 0]에 근사한 값이 반환될 것
    print(a, sess.run(tf.arg_max(a,1))) #arg_max를 통해 one-Hot Encoding 진행
    #위치를 리턴
    
    b=sess.run(hypothesis, feed_dict={X:[[1,3,4]]})
    print(b, sess.run(tf.arg_max(b,1)))
    
    c=sess.run(hypothesis, feed_dict={X:[[1,1,0]]})
    print(c, sess.run(tf.arg_max(c,1)))
    
    var_all=sess.run(hypothesis, feed_dict={X:[[1,11,7], [1,3,4], [1,1,0]]})
    print(var_all, sess.run(tf.arg_max(var_all,1)))