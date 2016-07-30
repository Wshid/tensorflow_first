import tensorflow as tf
import numpy as np

# 구현하기앙오오오오ㅗ 8:50부터

xy=np.loadtxt('train.txt', unpack=True) #unpack=true이 transpose하게 데이터를 가져옴 #4*3이 아닌 3*4
#x_data=xy[0:-1]
#y_data=xy[-1]

# 이작업이 4*3 -> 3*4 -> 4*2로 가져오는 작업

x_data=np.transpose(xy[0:-1]) #matmul을 정상적으로 하기 위해 변형 #4*2로 가져옴
y_data=np.reshape(xy[-1],(4,1)) #마지막 배열을 4*1형태로 가져옴
#row 4, column 1

X=tf.placeholder(tf.float32, [None,2])
Y=tf.placeholder(tf.float32, [None,1])
#########################################################
#W1=tf.Variable(tf.random_uniform([2,2], -1.0, 1.0)) #x1,x2의 입력값,
    #첫번째 2의 입력, 2개수로 출력
#W2=tf.Variable(tf.random_uniform([2,1], -1.0, 1.0))
    #첫번째 2개의 입력, 1개수로 출력
#b1=tf.Variable(tf.zeros([2]), name="Bias1")
#b2=tf.Variable(tf.zeros([1]), name="Bias2")

#L2=tf.sigmoid(tf.matmul(X,W1)+b1)
#hypothesis=tf.sigmoid(tf.matmul(L2,W2)+b2)
#################################################################

#W1=tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0)) #2개의 입력, 10개의 출력
#W2=tf.Variable(tf.random_uniform([10,1], -1.0 , 1.0)) #10개의 입력, 1개의 출력

#b1=tf.Variable(tf.zeros([10]), name="Bias1")

#####################################################################
#레이어를 더 두는것이 Deep
W1=tf.Variable(tf.random_uniform([2,5], -1.0, 1.0)) #2개의 입력 5개 출력
W2=tf.Variable(tf.random_uniform([5,4], -1.0, 1.0)) #5개의 입력 4개 출력
W3=tf.Variable(tf.random_uniform([4,1], -1.0, 1.0)) #4개 입력 1개 출력

b1=tf.Variable(tf.zeros([5]),name="Bias1")
b2=tf.Variable(tf.zeros([4]),name="Bias2")
b3=tf.Variable(tf.zeros([1]),name="Bias3") #name은 TensorBoard 사용시 노드 이름을 정하기 위함임

L2=tf.sigmoid(tf.matmul(X,W1)+b1) #Layer 증가 # classfication을 위해 sigmoid 함수 부분 적용
L3=tf.sigmoid(tf.matmul(L2,W2)+b2)
hypothesis=tf.sigmoid(tf.matmul(L3,W3)+b3)


cost=-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis)) #cost using cross-entropy

a=tf.Variable(0.01)
optimizer=tf.train.GradientDescentOptimizer(a)
train=optimizer.minimize(cost)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(200000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if(step%10000==0):
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2), sess.run(W3))
            
    correct_prediction=tf.equal(tf.floor(hypothesis+0.5),Y) #Test Model, 0.5보다 클경우 1, floor는 1또는 0 반환, Y는 실제 값
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float")) #Calculate Accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
    print("Accuracy:", accuracy.eval({X:x_data, Y:y_data}))

#2*10^6 실행시키니까 정확도 1.0 나옴