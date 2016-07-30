import tensorflow as tf

sess=tf.Session()

a=tf.constant(2) # 변수설정 방법이 특이하다, 하지만 이 역시 operation 취급
b=tf.constant(3)

c=a+b

print(sess.run(c)) # Session().run으로 해당 내용을 실행시킬 수 있다