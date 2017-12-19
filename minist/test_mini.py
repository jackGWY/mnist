from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
print("training data size:",mnist.train.num_examples)
print("validating data size:",mnist.validation.num_examples)
print("testing data size:",mnist.test.num_examples)
print("Example training data:",mnist.train.images[0])
print("example training data label: ",mnist.train.labels[0])