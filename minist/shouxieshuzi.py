"""
#!/usr/bin/python
#针对机器学习初学者的MNIST实验——源码和运行结果 
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  if i % 20 == 0:
    sys.stdout.write('.')
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print("")

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
in_units = 784 
h1_units = 300 
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1)) 
b1 = tf.Variable(tf.zeros([h1_units])) 
W2 = tf.Variable(tf.zeros([h1_units, 10])) 
b2 = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32) 


hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(3001):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    if i % 200 ==0:
		
        print(i, 'training_arruracy:', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, 
                             keep_prob: 1.0}))
print('final_accuracy:', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))