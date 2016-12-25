import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

def weight_var(shape):
	init_state=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init_state)

def bias_var(shape):
	init_state=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init_state)

x=tf.placeholder(tf.float32,shape=[None,784])            #placeholder for FEEDING in the x_trains
y_label=tf.placeholder(tf.float32,shape=[None,10])             #placeholder to feed in the labels
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')             #strides=[1,stride,stride,1],same padding->same dim output as i/p

def max_pool(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')   #ksize->dimension of i/p tensor=>1->no of images at      											     once,2*2,batch dim and channels are one because we are taking 											     max over 1 image and 1 channel ,not multi img or channel

def conv_net(x):
	#for conv x*W x->4d tensor [batch_size,height,width,channels]  , W->[height,width,i/p channel(same as channel of i/p tensor),o/p channel]

	#network blueprint:(conv->relu->maxpool)->(conv->relu->maxpool)->fc->o/p

	x=tf.reshape(x,[-1,28,28,1])                    #-1 to accomodate any  batch _size,1 for i/p channel
	w_conv1=weight_var([5,5,1,32])
	b_conv1=bias_var([32])
	a_conv1=tf.nn.relu(tf.add(conv2d(x,w_conv1),b_conv1))
	a_pool1=max_pool(a_conv1)
	
	w_conv2=weight_var([5,5,32,64])
	b_conv2=bias_var([64])
	a_conv2=tf.nn.relu(tf.add(conv2d(a_pool1,w_conv2),b_conv2))
	a_pool1=max_pool(a_conv2)

	tmp=tf.reshape(a_pool1,[-1,7*7*64])
	w_fc1=weight_var([7*7*64,1024])
	b_fc1=bias_var([1024])
	z_1=tf.add(tf.matmul(tmp,w_fc1),b_fc1)
	a_1=tf.nn.relu(z_1)
	

	
    	h_fc1_drop = tf.nn.dropout(a_1, keep_prob)
    	
	W_fc2 = weight_var([1024, 10])
   	b_fc2 = bias_var([10])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    	return y_conv
	

def minimize_loss():
  
	y_conv=conv_net(x)
	
	cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,y_label))
	optimizer=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
	
	correctness=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_label,1))
	accuracy=tf.reduce_mean(tf.cast(correctness,'float'))
	
        with tf.Session() as sess:
	
		sess.run(tf.initialize_all_variables())
		n_iter=20000
		for i in range(n_iter):
  			batch = mnist.train.next_batch(50)
  			
  			if i%100 == 0:
    				train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_label: batch[1], keep_prob: 1.0})
    				print("step %d, training accuracy %g"%(i, train_accuracy))
  			
  			optimizer.run(feed_dict={x: batch[0], y_label: batch[1], keep_prob: 0.5})

		print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1.0}))

minimize_loss()



	
	
