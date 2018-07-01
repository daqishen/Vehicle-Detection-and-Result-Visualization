# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 01:59:52 2018

@author: qiyue
"""

import tensorflow as tf
import cv2
sess = tf.Session()

a1 = tf.constant(5.0)
b = tf.constant(6.0)
c = a1 * b

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print(sess.run(c))
    #%%
    
y = tf.Variable(1)
b = tf.identity(y)
tf.global_variables_initializer().run()
print(sess.run(b,feed_dict={y:3})) #使用3 替换掉
#tf.Variable(1)的输出结果，所以打印出来3 
#feed_dict{y.name:3} 和上面写法等价

print(sess.run(b))  #由于feed只在调用他的方法范围内有效，所以这个打印的结果是 1
#%%
a = tf.placeholder("float")
b = tf.placeholder("float")
c = tf.constant(6.0)
d = tf.multiply(a, b)
y = tf.multiply(d, c)
print (sess.run(y, feed_dict={a: 3, b: 3}))
#%%
a = tf.constant([10, 20])
b = tf.constant([1.0, 2.0])
v = sess.run(a)
v1 = sess.run([a, b])
v2 = sess.run({'k1': [a,b], 'k2': [b, a]})
v3 = {'k3':[],'k4':[]}

#%%
x = tf.placeholder(tf.float32, (None,), 'x')
y = tf.reduce_sum(x)
a = sess.run(y, {x: [1, 2, 3]})
sess.run(y, {'x:0': [1, 2, 3]})


#%%
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)
#%%
inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


with tf.Session() as sess:
  sess.run(init_op)
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)


with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())


#%%
import tensorflow as tf
a = tf.constant([10, 20])
sess = tf.Session()
sess.run(a)







im = cv2.imread(im_file)
cv2.imshow('filtered image', im)	
#
#show the image 
cv2.waitKey(0)                      # 0 means wait indefinitely for a keystroke. When a key is pressed, the program proceeds
cv2.destroyAllWindows()             # Close the image window













