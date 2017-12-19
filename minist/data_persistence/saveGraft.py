import tensorflow as tf
from tensorflow.python.framework import graph_util


def Save_Graft():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v1")
    result = v1 + v2
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
        # 这个方法要自己创建一个文件夹
        with tf.gfile.GFile("E:/python/tensorflow_learning/minist/data_persistence/model3/combined_modle.pb",
                            "wb") as f:
            f.write(output_graph_def.SerializeToString())

#Save_Graft()

def loadGraft():
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        model_filename="E:/python/tensorflow_learning/minist/data_persistence/model3/combined_modle.pb"
        with gfile.FastGFile(model_filename,'rb') as f:
            #读取保存的文件模型，解析成graph protocol buffer
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
        result=tf.import_graph_def(graph_def,return_elements=['add:0'])
        print(sess.run(result))
loadGraft()


