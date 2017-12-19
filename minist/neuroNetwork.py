import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE=784#输入曾节点数目
OUTPUT_NODE=10#输出层节点数目

LAYYER1_NODE=500#隐藏层个数
#LAYER2_NODE=500
BATCH_SIZE=100#一个batch 中训练数据个数

LEARNING_RATE_BASE=0.8#基础学习效率
LEARNING_RATE_DECAY=0.99#学习率的衰减率

REGULARIZATION_RATE=0.0001#描述模型复杂度的正则划项在损失函数中的系数
TRAINNING_STEPS=10000#训练轮数
MOVING_AVERAGE_DECAY=0.99#滑动平均衰减率

#给定神经网络的输入和所有参数。计算神经网络向前传播
def inference(input_tensor,avg_class,weights1,biase1,weights2,biase2):
    #当没有提供滑动平均时   直接使用参数当前的取值
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biase1)
        return tf.matmul(layer1,weights2)+biase2
    else:
        layer1=tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biase1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biase2)

# def ingerence(input_tensor,avg_class,reuse=False):
#     if avg_class==None:
#         with tf.variable_scope('layer1',reuse=reuse):
#             weights1=tf.get_variable("weights1",[INPUT_NODE,LAYYER1_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
#             biases1=tf.get_variable("biases1",[LAYYER1_NODE],initializer=tf.constant_initializer(0.1))


def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    #生产隐藏层参数
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYYER1_NODE],stddev=0.1))#stddev 偏差
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYYER1_NODE]))
    #生产输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYYER1_NODE,OUTPUT_NODE ], stddev=0.1) ) # stddev 偏差
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y=inference(x,None,weights1,biases1,weights2,biases2)
    #训练轮数
    global_step=tf.Variable(0,trainable=False)
    #初始化滑动平均的类
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_averages_op=variable_averages.apply(tf.trainable_variables())

    average_y=inference(x,variable_averages,weights1,biases1,weights2,biases2)

    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))#dims=1  行上最大的数的下标
    #计算当前batch中的所有交叉平均值
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    #计算L2正则划损失函数
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则划损失，一般计算只计算权值的损失，不计算B
    regularization=regularizer(weights1)+regularizer(weights2)
    #总损失等于交叉熵损失和正则划损失的和
    loss=cross_entropy_mean+regularization
    #设置衰减学习率
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,#当前迭代轮数
        mnist.train.num_examples/BATCH_SIZE,#过完所有训练数据需要迭代的次数
        LEARNING_RATE_DECAY#学习衰减速率
    )
    train_step=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
        .minimize(loss,global_step)
    #通过反响传播跟新神经网络中的参数，又要跟新每一个参数的滑动平均值
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #将bool值变为实数，计算平均，平均值就是正确率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #准备验证数据，
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        #测试数据
        test_feed={x:mnist.test.images,y_:mnist.test.labels}

        for i in range(TRAINNING_STEPS):
            if i%1000==0:#每1000轮输出验证数据集的测试结果
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                test_acc=sess.run(accuracy,feed_dict=test_feed)
                print("afer %d training step(s),validation accuracy "
                      "using average model is %g ,test accuracy using average model is %g"
                      % (i,validate_acc,test_acc))
            #产生新一轮batch训练数据，
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test accuracy using average "
              "modle is %g" % (TRAINNING_STEPS,test_acc))
def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)
if __name__=="__main__":
    tf.app.run()

















