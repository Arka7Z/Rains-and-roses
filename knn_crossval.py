from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
#the iris data set will be taken as an example
 
iris=load_iris()
x=iris.data
y=iris.train
 
k_range=range(1,41)
k_scores=[]    #to store the accuracy score for the range of k
for k in k_range:
  knn=KNeighborsClassifier(n_neighbors=k)
  score=cross_val_score(knn,x,y,scoring='accuracy')
  k_scores.append(scores.mean())
 
plt.plot(k_range,k_scores)
plt.xlabel('Value of k')
plt.ylabel('Accuracy Score')
