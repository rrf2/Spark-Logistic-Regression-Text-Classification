# Spark-Logistic-Regression-Text-Classification
This is a logistic regression text classifier implemented in Spark using Python. Specifically, it determines whether a given piece of text is a Wikipedia article or an Australian court case. It uses each document's TF-IDF (Term Frequency - Inverse Document Frequency) vector on the top 20000 words in the corpus as the feature vector. L2 regularization is also used to prevent overfitting. The model has an F1 score of 0.952.
