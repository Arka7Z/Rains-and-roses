import tensorflow as tf
import pandas as pd
import numpy as np

data=pd.read_csv('/home/amanpurwar/iris.csv',names=['s_len','s_width','p_length','p_width','class'])

x=tf.placeholder(tf.float32,shape=[None,4])
y=tf.placeholder(tf.float32,shape=[None,3])



#converting pd into one hot encoding

setosa=np.asarray([1,0,0])
versicolor=np.asarray([0,1,0])
virginica=np.asarray([0,0,1])

#data['class']=data['class'].map({'Iris-setosa': setosa, 'Iris-versicolor': versicolor,'Iris-virginica':virginica})


species= list(data['class'].unique())
data['One-hot'] = data['class'].map(lambda x: np.eye(len(species))[species.index(x)] )



#shuffle data

data=data.iloc[np.random.permutation(len(data))]
data=data.reset_index(drop=True)


#train_test splitting

x_train=data.ix[0:len(data)-40,['s_len','s_width','p_length','p_width']]
y_train=data.ix[0:len(data)-40,['class','One-hot']]
trainSet=data.ix[0:len(data)-40]

x_test=data.ix[len(data)-40:,['s_len','s_width','p_length','p_width']]
y_test=data.ix[len(data)-40:,['class','One-hot']]




def weight_var(shape):
	init_state=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init_state)
	
def bias_var(shape):
	init_state=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init_state)
	
def runANN(x):
	n_hl1=400
	n_hl2=450
	n_hl3=400
	n_class=3   #3 way classification
	
	weight1=weight_var([4,n_hl1])
	bias1=bias_var([n_hl1])
	a_2=tf.nn.relu(tf.add(tf.matmul(x,weight1),bias1))
	
	weight2=weight_var([n_hl1,n_hl2])
	bias2=bias_var([n_hl2])
	a_3=tf.nn.relu(tf.add(tf.matmul(a_2,weight2),bias2))
	
	weight3=weight_var([n_hl2,n_hl3])
	bias3=bias_var([n_hl3])
	a_4=tf.nn.relu(tf.add(tf.matmul(a_3,weight3),bias3))
	
	weight_final=weight_var([n_hl3,n_class])
	bias_final=bias_var([n_class])
	y_pred=tf.nn.relu(tf.add(tf.matmul(a_4,weight_final),bias_final))
	
	return y_pred
	
def minimize_loss():
	y_pred=runANN(x)
	cross_entropy=tf.nn.softmax_cross_entropy_with_logits(y_pred,y)
	optimizer=tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
	correct=tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
	accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
	
	with tf.Session() as sess:
		keys=['s_len','s_width','p_length','p_width']
		sess.run(tf.global_variables_initializer())
		n_iter=20000
		for i in range(n_iter):
			train=trainSet.sample(50)
			
			#sess.run(optimizer,feed_dict={x:x_train,y:[t for t in y_train['One-hot'].as_matrix()]})
			sess.run(optimizer,feed_dict={x:[t for t in train[keys].values],y:[t for t in train['One-hot'].as_matrix()]})
			
			
		acc=sess.run(accuracy,feed_dict={x:x_test,y:[t for t in y_test['One-hot'].as_matrix()]})
		print("Accuracy is:")
		print(acc)
	
	
minimize_loss()








