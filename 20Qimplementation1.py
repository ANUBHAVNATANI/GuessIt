
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


from sklearn.tree import DecisionTreeClassifier


# In[4]:


from sklearn import tree


# In[11]:


trainData=pd.read_csv("TrainData.csv",sep=',')


# In[12]:


print("Dataset Length::",len(trainData))


# In[13]:


print("Dataset Shape::",trainData.shape)


# In[14]:


#first 5 rows of dataset
trainData.head()


# In[15]:


Clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=None)


# In[16]:


X=trainData.values[:,1:24]
print(X)


# In[18]:


Y=trainData.values[:,0]
print(Y)


# In[19]:


Clf_gini.fit(X,Y)


# In[22]:


#now we are making decision tree clf with information gain
clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=None)


# In[23]:


clf_entropy.fit(X,Y)


# In[25]:


#we have trained our both models now its time to predictions
Clf_gini.predict([[0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1]])


# In[26]:


#we got the correct out-put but it is assuming all the answers are correct.
#when we change the input a little bit then see what happens
Clf_gini.predict([[0,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,1,1,1,0,0,0,1]])


# In[28]:


#then it also guess garlic now if the main input is wrong
Clf_gini.predict([[1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1]])


# In[29]:


#we can now see that it is the first one on the top of the decision tree
clf_entropy.predict([[0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1]])


# In[30]:


#we can now see that our entropy classifier is also giving the correct answers
clf_entropy.predict([[1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1]])


# In[31]:


#both the classifier seems to work the same we can check deeply by graphing them 
from sklearn.tree import export_graphviz


# In[33]:


from sklearn.externals.six import StringIO


# In[37]:


from IPython.display import Image
dot_data = StringIO()


# In[38]:


export_graphviz(Clf_gini, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)


# In[41]:


import pydotplus


# In[44]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[50]:


dot_data_two=StringIO()
export_graphviz(clf_entropy, out_file=dot_data_two,  
                filled=True, rounded=True,
                special_characters=True)


# In[51]:


graph = pydotplus.graph_from_dot_data(dot_data_two.getvalue())  
Image(graph.create_png())


# In[52]:


#now we have to repeat the process by changing the paramerters
#from this info we determined the entropy method resulted in minimum depth and more versitile


# In[53]:


new_clf=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3)


# In[54]:


new_clf.fit(X,Y)


# In[55]:


new_clf.predict([[0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1]])


# In[56]:


#As we can see that it is giving the wrong answers when tree depth is 3
new_clf=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=5)


# In[57]:


new_clf.fit(X,Y)


# In[58]:


new_clf.predict([[0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1]])


# In[60]:


#As we have said depth to be 5 it is giving correct result on an simple input now test it on some difficult input
new_clf.predict([[0,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,1,1,1,0,0,0,1]])


# In[61]:


#it also strated giving the wrong ansewer 
new_clf.predict([[1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1]])


# In[62]:


#it identifies garlic as fruit thats not good it should be the basis of classification so here our model preform badly
dot_data_three=StringIO()
export_graphviz(new_clf, out_file=dot_data_three,  
                filled=True, rounded=True,
                special_characters=True)


# In[63]:


graph = pydotplus.graph_from_dot_data(dot_data_three.getvalue())  
Image(graph.create_png())


# In[ ]:


#from all these observation we see that for these kind of things we should clearly seperate all the leaves
#seperating all the leaves would help in the good model formation

