import tensorflow as tf 
import numpy as n 

w = tf.Variable([3.0])
b = tf.Variable([-3.0])
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
lin = w*x+b
loss = tf.reduce_sum(tf.square(lin-y))

opt = tf.train.GradientDescentOptimizer(0.01)
train = opt.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	sess.run(train,{x:[1,2,3,4],y:[5,6,7,8]})

print(sess.run([w,b]))
print(sess.run(lin,{x:[1,2,3,4]}))
print(sess.run(loss,{x:[1,2,3,4],y:[5,6,7,8]}))
# print(n1,n2)
# print(sess.run(c,{a:[3,4],b:[5,7]}))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,w)+b
loss = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)

# loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))


opt = tf.train.GradientDescentOptimizer(0.5)
trainstep = opt.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for _ in range(1000):
	batch_x,batch_y = mnist.train.next_batch(100)
	sess.run(trainstep,{x:batch_x,y_:batch_y})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

