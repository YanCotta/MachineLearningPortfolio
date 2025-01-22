# Natural Language Processing - Sentiment Analysis - Multiple Models 


# Importing the libraries (numpy, matplotlib, pandas, nltk, re, sklearn)
import numpy as np #mathematical library
import matplotlib.pyplot as plt #plotting library
import pandas as pd #import and manage datasets
import re #regular expressions
import nltk #natural language toolkit

# Importing the dataset (Tsv file)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #the delimiter is tab, quoting = 3 ignores double quotes

# Cleaning the texts (data preprocessing)
nltk.download('stopwords') #download the stopword
from nltk.corpus import stopwords #import the stopwords
from nltk.stem.porter import PorterStemmer #import the stemmer
corpus = [] #initialize the corpus
for i in range (0, 1000): #iterate through the dataset
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #remove all characters except letters
    review = review.lower() #convert all letters to lowercase
    review = review.split() #split the words
    ps = PorterStemmer() #initialize the stemmer
    all_stopwords = stopwords.words('english') #get all the stopwords
    all_stopwords.remove('not') #remove the word 'not' from the stopwords
    review = [ps.stem(word) for word in review if not word in set (stopwords.words('english'))] #remove the stopwords
    review = ' '.join(review) #join the words
    corpus.append(review) #append the review to the corpus

# Creating the Bag of Words model (Tokenization)
from sklearn.feature_extraction.text import CountVectorizer #import the CountVectorizer
cv = CountVectorizer (max_features = 1500) #initialize the CountVectorizer
X = cv.fit_transform(corpus).toarray() #create the sparse matrix
y = dataset.iloc[:, -1].values #create the dependent variable

# Splitting the dataset into the Training set and Test set (80% training, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #import the train_test_split


# Training the Naive Bayes model on the Training set 
from sklearn.naive_bayes import GaussianNB #import the GaussianNB
classifier = GaussianNB() #initialize the GaussianNB
classifier.fit(X_train, y_train) #fit the model to the training


# Predicting the Test set results (Naive Bayes)
y_pred = classifier.predict(X_test) #predict the test set
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) #display the predictions and the actual values

# Making the Confusion Matrix (Naive Bayes)
from sklearn.metrics import confusion_matrix, accuracy_score #import the confusion_matrix and accuracy_score
cm = confusion_matrix(y_test, y_pred) #create the confusion matrix
print(cm) #display the confusion matrix
accuracy_score(y_test, y_pred) #display the accuracy score