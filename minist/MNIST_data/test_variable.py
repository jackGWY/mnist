import tensorflow as tf
v1=tf.get_variable("v",[1])
print(v1.name)
with tf.variable_scope("foo"):
    v2=tf.get_variable("v",[1])
    print(v2.name)
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3=tf.get_variable("v",[1])
        print(v3.name)
    v4=tf.get_variable("v1",[1])
    print(v4.name)
with tf.variable_scope("",reuse=True):#reuse=true   这样tf.getvariable  函数就可以获取声明的变量
    v5=tf.get_variable("foo/bar/v",[1])
    print(v5==v3)
    v6=tf.get_variable("foo/v1",[1])
    print(v6==v4)