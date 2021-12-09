#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

newstuff = "\n\n--------------------------------\n\n"
ns=newstuff

input1 = tf.constant(2.0)
input2 = tf.constant(5.0)
print(input1, input2)
print(newstuff)
#sess = tf.compat.v1.Session()
sess = tf.Session()
print(sess.run([input1, input2]))
add_node = tf.add(input1, input2)
print(add_node)
print(sess.run(add_node))
print(newstuff)
p1 = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)
add_ph = p1 + p2
print(sess.run(add_ph, {p1: 2, p2: 5}))
print(sess.run(add_ph, {p1: 1.2, p2: 3.5}))
print(sess.run(add_ph, {p1: [1, 2], p2: [5, 8]}))
print(newstuff)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
a = tf.Variable([1], dtype=tf.float32)
b = tf.Variable([-2], dtype=tf.float32)
linear_model = a*x + b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[0,1,2,3,4,5]}))
print(newstuff)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
feed_dict = {
  x:[0,1,2,3,4,5],
  y:[-1,-.5,0,.5,1,1.5] }
print(sess.run(loss, feed_dict))
assignA = tf.assign(a, [.25])
assignB = tf.assign(b, [0])
sess.run([assignA, assignB])
print(sess.run(loss, feed_dict))
assignA = tf.assign(a, [.5])
assignB = tf.assign(b, [-1])
sess.run([assignA, assignB])
print(sess.run(loss, feed_dict))
print(ns)
