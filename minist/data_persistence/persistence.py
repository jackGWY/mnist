import tensorflow as tf
v1=tf.Variable(tf.constant(1.0,shape=[1],name="v1"))
v2=tf.Variable(tf.constant(2.0,shape=[1],name="v2"))
result=v1+v2

init_op=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    # sess.run(init_op)
    # saver.save(sess,"E:/python/tensorflow_learning/minist/data_persistence/model/model.ckpt")
    saver.restore(sess,"E:/python/tensorflow_learning/minist/data_persistence/model/model.ckpt")
    print(sess.run(result))

saver2=tf.train.import_meta_graph("E:/python/tensorflow_learning/minist/data_persistence/model/model.ckpt.meta")
with tf.Session() as sess:
    saver2.restore(sess,"E:/python/tensorflow_learning/minist/data_persistence/model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

