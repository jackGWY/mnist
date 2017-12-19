#encoding:utf-8
import tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=10
IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10
#第一层卷积层的尺度和深度
CONV1_DEP=32
CONV1_SIZE=5
#第2层卷积层的尺度和深度
CONV2_DEP=64
CONV2_SIZE=5
#全链接层节点个数
FC_SIZE=52
#定义向前传播的过程

def inference(input_tensor,train,regularizer):
    #第一层向前传播，输入：28*28*1，过滤器：5*5*32 输出：28*28*32
    with tf.variable_scope("layer1-conv1"):
        conv1_weights=tf.get_variable("weights",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEP],initializer=tf.truncated_normal_initializer(stddev=0.1))

    conv1_biases=tf.get_variable("bias",[CONV1_DEP],initializer=tf.constant_initializer(0.0))
    conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
    relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #第二层池化层向前传播，输入28*28*32，过滤器[1,2,2,1]来输出14*14*32
    with tf.name_scope("layer2-pool1"):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #第三层卷积层，输入14*14*32，输出14*14*32，因为加了padding
    with tf.variable_scope("layer3-conv2"):
        conv2_weights=tf.get_variable("weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEP,CONV2_DEP],initializer=tf.truncated_normal_initializer(stddev=0.1));
        conv2_biases=tf.get_variable("bias",[CONV2_DEP],initializer=tf.constant_initializer(0.0))
        #使用边长5，深度64的过滤器，过滤移动步长为1，
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding="SAME")
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    #实现第四层池化向前,输入14*14*32，输出7*7*64
    with tf.name_scope("layer4-pool2"):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    pool_shape=pool2.get_shape().as_list()
    #拉直后向量的长度
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    #将第四层输出变为一个batch向量
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])
    with tf.variable_scope("layer5-fc1"):
        fcl_weights=tf.get_variable(
            "weight",[nodes,FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fcl_weights))
        fcl_biases=tf.get_variable("bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fcl=tf.nn.relu(tf.matmul(reshaped,fcl_weights)+fcl_biases)
        #dropout  训练是随机将部分节点输出改为0，避免过拟合。
        if train:fcl=tf.nn.dropout(fcl,0.5)
    with tf.variable_scope("layer6-fc2"):
        fc2_weights=tf.get_variable("weight",[FC_SIZE,NUM_LABELS],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection("losses",regularizer(fc2_weights))
        fc2_biases=tf.get_variable("bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fcl,fc2_weights)+fc2_biases
    return logit




