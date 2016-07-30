import tensorflow as tf

a=tf.constant(2) # 'a' variable is not constant number, but it is operation oneself
b=tf.constant(3)

with tf.Session() as sess:
    c=a+b # return tensor type, it is operation not variable.
    print(c)
    print(sess.run(c)) # can view return value of operations on tensorflow