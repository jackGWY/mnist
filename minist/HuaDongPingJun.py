import tensorflow as tf
v1=tf.Variable(0,dtype=tf.float32)
step=tf.Variable(0,trainable=False)
#定义一个滑动平均的类，初始化衰减率0.99
ema=tf.train.ExponentialMovingAverage(0.99,step)
maintain_averages_op=ema.apply([v1])

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    #通过ema.average(v1)获取滑动平均的取值
    print(sess.run([v1,ema.average(v1)]))
    #将v1的值设置5
    sess.run(tf.assign(v1,5))

    sess.run(maintain_averages_op)#跟新v1的滑动品军值，min{},shadom_variable
    print(sess.run([v1,ema.average(v1)]))

    sess.run(tf.assign(step,10000))
    sess.run(tf.assign(v1,10))

    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))

    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))