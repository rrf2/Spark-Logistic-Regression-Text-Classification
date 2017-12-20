# Spark-Logistic-Regression-Text-Classification
This is a logistic regression text classifier implemented in Spark using Python. Specifically, it determines whether a given piece of text is a Wikipedia article or an Australian court case. It uses each document's TF-IDF (Term Frequency - Inverse Document Frequency) vector on the top 20000 words in the corpus as feature vectors. L2 regularization is also used to prevent overfitting. The model has an F1 score of 0.952.

Here is the training and testing data used:

Training data: https://www.dropbox.com/s/5kyhidi10qa74t1/training.txt?dl=0

Testing data: https://www.dropbox.com/s/5f7y7qus2np84xj/testing.txt?dl=0
