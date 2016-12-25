import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

x=tf.placeholder(tf.float32,shape=[None,784])
y=tf.placeholder(tf.float32,shape=[None,10])

def weight_var(shape):
	init_state=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init_state)
	
def bias_var(shape):
	init_state=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init_state)

def mlp(x):
	#layer 1
	n_hl_1=450
	n_hl_2=500
	n_hl_3=450
	n_class=10  #mnist->10 way classification
	
	weight1=weight_var([784,n_hl_1])
	bias1=bias_var([n_hl_1])
	a_2=tf.nn.relu(tf.add(tf.matmul(x,weight1),bias1))           #broadcasting takes care of the dimensional difference
	
	weight2=weight_var([n_hl_1,n_hl_2])
	bias2=bias_var([n_hl_2])
	a_3=tf.nn.relu(tf.add(tf.matmul(a_2,weight2),bias2))
	
	weight3=weight_var([n_hl_2,n_hl_3])
	bias3=bias_var([n_hl_3])
	a_4=tf.nn.relu(tf.add(tf.matmul(a_3,weight3),bias3))
	
	#final classification layer of 10 nodes for 10 way classification of mnist
	
	weight_final=weight_var([n_hl_3,n_class])
	bias_final=bias_var([n_class])
	y_pred=tf.nn.relu(tf.add(tf.matmul(a_4,weight_final),bias_final))
	
	return y_pred
	
	
def min_loss():
	
	y_pred=mlp(x)
	cross_entropy=tf.nn.softmax_cross_entropy_with_logits(y_pred,y)
	optimizer=tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
	correct=tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
	accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		n_iter=20000
		for i in range(n_iter):
			x_epoch,y_epoch=mnist.train.next_batch(100)
			sess.run(optimizer,feed_dict={x:x_epoch,y:y_epoch})
			
		acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
		print("Accuracy is:")
		print(acc)
	
	
min_loss()
