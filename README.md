##candidate elimination algorithm    

The candidate elimination algorithm incrementally builds the version space given a hypothesis space H and a set E of examples. 
The examples are added one by one; each example possibly shrinks the version space by removing the hypotheses that are inconsistent with the example. 
The candidate elimination algorithm does this by updating the general and specific boundary for each new example. 

You can consider this as an extended form of Find-S algorithm.
Consider both positive and negative examples.
Actually, positive examples are used here as Find-S algorithm (Basically they are generalizing from the specification).
While the negative example is specified from generalize form.
Terms Used:  

Concept learning: Concept learning is basically learning task of the machine (Learn by Train data)
General Hypothesis: Not Specifying features to learn the machine.
G = {‘?’, ‘?’,’?’,’?’…}: Number of attributes
Specific Hypothesis: Specifying features to learn machine (Specific feature)
S= {‘pi’,’pi’,’pi’…}: Number of pi depends on number of attributes.
Version Space: It is intermediate of general hypothesis and Specific hypothesis. It not only just written one hypothesis but
a set of all possible hypothesis based on training data-set.

Algorithm:

Step1: Load Data set
Step2: Initialize General Hypothesis  and Specific  Hypothesis.
Step3: For each training example  
Step4: If example is positive example  
          if attribute_value == hypothesis_value:
             Do nothing  
          else:
             replace attribute value with '?' (Basically generalizing it)
Step5: If example is Negative example  
          Make generalize hypothesis more specific.
          
          
          
##decison tree algorithm 

Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for
solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, 
branches represent the decision rules and each leaf node represents the outcome.
In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node.
Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.
The decisions or the test are performed on the basis of features of the given dataset.
It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.
It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.


##find- S algorithm

Find-S algorithm is a basic concept learning algorithm in machine learning. Find-S algorithm finds the most specific hypothesis that fits all the positive examples. 
We have to note here that the algorithm considers only those positive training example. Find-S algorithm starts with the most specific hypothesis and generalizes this 
hypothesis each time it fails to classify an observed positive training data. Hence, Find-S algorithm moves from the most specific hypothesis to the most general hypothesis.
Steps Involved In Find-S 

Important Representation :

1. ? indicates that any value is acceptable for the attribute.
2. specify a single required value ( e.g., Cold ) for the attribute.
3. ϕindicates that no value is acceptable.
4. The most general hypothesis is represented by: {?, ?, ?, ?, ?, ?}
5. The most specific hypothesis is represented by : {ϕ, ϕ, ϕ, ϕ, ϕ, ϕ}

Start with the most specific hypothesis.
h = {ϕ, ϕ, ϕ, ϕ, ϕ, ϕ}
1. Take the next example and if it is negative, then no changes occur to the hypothesis.
2. If the example is positive and we find that our initial hypothesis is too specific then we update our current hypothesis to general condition.
3. Keep repeating the above steps till all the training examples are complete.
4. After we have completed all the training examples we will have the final hypothesis when can used to classify the new examples.


 ##KNN algorithm

1. K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.
2. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.
3. K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily
   classified into a well suite category by using K- NN algorithm.
5. K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.
6. K-NN is a non-parametric algorithm, which means it does not make any assumption on underlying data.
7. It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, 
  it performs an action on the dataset.
8. KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.


How does K-NN work?
The K-NN working can be explained on the basis of the below algorithm:

Step-1: Select the number K of the neighbors
Step-2: Calculate the Euclidean distance of K number of neighbors
Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
Step-4: Among these k neighbors, count the number of the data points in each category.
Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
Step-6: Our model is ready.


##linear regression

Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value 
based on independent variables. It is mostly used for finding out the relationship between variables and forecasting.
Different regression models differ based on – the kind of relationship between dependent and independent variables, they are considering and the number of 
independent variables being used.


Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). 
So, this regression technique finds out a linear relationship between x (input) and y(output). Hence, the name is Linear Regression.
In the figure above, X (input) is the work experience and Y (output) is the salary of a person. The regression line is the best fit line for our model.

Regression in machine learning consists of mathematical methods that allow data scientists to predict a continuous outcome (y) based on the value of 
one or more predictor variables (x). Linear regression is probably the most popular form of regression analysis because of its ease-of-use in predicting and forecasting.


 ##multi regression algorithm
 
Multiple regression is a machine learning algorithm to predict a dependent variable with two or more predictors. Multiple regression has numerous
real-world applications in three problem domains: examining relationships between variables, making numerical predictions and time series forecasting.

Below are the main steps of deploying the MLR model:

1. Data Pre-processing Steps
2. Fitting the MLR model to the training set
3. Predicting the result of the test set


 ##simple neural network(perception)
 
Neural networks are a class of machine learning algorithms used to model complex patterns in datasets using multiple hidden layers and non-linear activation functions. 
... Neural networks are trained iteratively using optimization techniques like gradient descent.
The structure of the human brain inspires a Neural Network. It is essentially a Machine Learning model (more precisely, Deep Learning) that is used in unsupervised learning. 
A Neural Network is a web of interconnected entities known as nodes wherein each node is responsible for a simple computation.


 ##k-meams algorithm

K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.
It allows us to cluster the data into different groups and a convenient way to discover the categories of groups in the unlabeled dataset on its own without
the need for any training.

It is a centroid-based algorithm, where each cluster is associated with a centroid. The main aim of this algorithm is to minimize the sum of distances between 
the data point and their corresponding clusters.

The algorithm takes the unlabeled dataset as input, divides the dataset into k-number of clusters, and repeats the process until it does not find the best clusters.
The value of k should be predetermined in this algorithm.

The k-means clustering algorithm mainly performs two tasks:

1. Determines the best value for K center points or centroids by an iterative process.
2. Assigns each data point to its closest k-center. Those data points which are near to the particular k-center, create a cluster.
