import tensorflow as tf
v=tf.Variable(0,dtype=tf.float32,name='v')
for variables in tf.all_variables():
    pass
    #print(variables.name)
ema=tf.train.ExponentialMovingAverage(0.99)
maintain_average_op=ema.apply(tf.all_variables())
for variables in tf.all_variables():
    print(variables.name)

saver=tf.train.Saver()
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v,10))
    sess.run(maintain_average_op)
    saver.save(sess,"E:/python/tensorflow_learning/minist/data_persistence/model2/model2.ckpt")
    print(sess.run([v,ema.average(v)]))
