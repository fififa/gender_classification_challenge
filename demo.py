

##Results:  With the data given, Decision Tree, KN and MLP classifier give the most accurate predictions.

# CHALLENGE - create 3 more classifiers...
# 1 - DecisionTreeClassifer
# 2-  KNeighborsClassifier
# 3 - Linear SVC (support vector classification) #capable of multi-class classification on a dataset.
#4 (new for me) - MLP Classifier
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

test = [[190, 70, 40]]; #Test list
# CHALLENGE #1 - Tree classifier trained  on our data
from sklearn import tree
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(X, Y)
prediction1 = clf1.predict(a) #Predicts male first round; female second round.s
proba1 = clf1.predict_proba(a) #The probabilities of prediction
#When I changed the shoe size to smaller ones, like 38, 37 - it predicts female. Same ting when I really reduce the height.
#When I reduce the height a lot, like 135 but increase the shoe size to 42, it predicts male... 
#so shoe size has a lot of weight.

# CHALLENGE #2 - KNeighbors classifier trained  on our data
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors=4) 
clf2 = clf2.fit(X, Y) 
KNeighborsClassifier(...)
prediction2 = clf2.predict(a)  #Predicts male;
proba2 = clf2.predict_proba(a) #The probabilities of prediction

#When I change the shoe size to 37, all else constant, still get male
#Shoe 37 and leight of 160, still get male. So it's very different from the tree decision. 
#I suspect the tree decision performs better.

# CHALLENGE #3 - Linear SVC classifier trained  on our data
from sklearn import svm
clf3 = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf3 = clf3.fit(X, Y)  
#This is how we've set up our parameters. Credit to Analytics Vidhya who explained
#How do you know which of these paramater values is the best? 
#How do you know which parameter is needed?
prediction3 = clf3.predict(a) #Predicts male;
#When I make shoe size to smaller, all else constant, it remains male.
#When I make height smaller, all else constant, it remains male. 
#When I make both height and shoe size smaller, I still get male, jesus!
#When I change C to the parameters - C and gamma, I stay get the male results. 
proba3 = clf3.predict_proba(a) #The probabilities of prediction

#################
#OTHER SOLUTIONS#
#################
#Credits to jeev20: 
#https://github.com/llSourcell/gender_classification_challenge/pull/2/commits/8d76bfb36c45d63ccee8ae161207f9dff4f6e328
#Something called GaussianProcessClassifier() - did yiel such good results!

# CHALLENGE #4 - MLP Classifier fron scikit learn (didn't know about it! Yielded good results)
clf4 = MLPClassifier(learning_rate='constant', learning_rate_init=0.001,)
#How did they figure out the parameters?
clf4 = clf4.fit(X, Y)
prediction4 = clf4.predict(a)
proba4 = clf4.predict_proba(a) #The probabilities of prediction

# FINAL CHALLENGE compare their results and print the best one!
print "Decision Tree - DT -  Classifier test data {} is predicted as {} with probability of {}".format(test, prediction1, proba1)

print "SVM Classifier test data {} is predicted as {} with probability of {}".format(test, prediction3, proba3)

print "KNeighbours classifier test data {} is predicted as {} with probability of {}".format(test,  prediction2, proba2)

print "MPL Classifier test data {} is predicted as {} with probability of {}".format(test, prediction4, proba4 )

##FURTHER QUESTIONS
#Which model performs better?
#2: #How do you know how many neighbors is good?
#When I use only 2 or 4 neighbours, it predicts the male as female! But with 3 or any digit after 6 it does it correctly.

##WHAT I LEARNT
#3: SVM models.
# svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
#Parameters: kernel, gamma and C matter the most.
#kernel- depends on the plane (way) we wanna use to separate the classes. 
#kernel - One could make it a linear separation (more errors) or a curvy one (different options like poly or rbf)
#kernel - we'll use rbf, for more curvy and less performance errors.
#Gamma - it modifies how tight or open the "territory" of my prediction for a certain class (or category) is. 
#Gamma - The smaller, the wider the territory. The bigger, the more narrow it is.
#C - it determines the balance between precision and logical grouping. 
#C - the higher it is, the more precise it gets at the expense of bad grouping. That's overfitting.
