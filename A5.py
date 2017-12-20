import re
import numpy as np

### LOADING AND PREPARING DATA

# load up all of the 19997 documents in the corpus
corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# (docID, text)
docAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

docAndCourtCase = docAndText.map(lambda x: (x[0], 1 if x[0][0:2] == 'AU' else 0))

regex = re.compile('[^a-zA-Z]')

# (docID, ["word1", "word2", "word3", ...])
docAndListOfWords = docAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# to ("word1", 1) ("word2", 1)...
allWords = docAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# (word, count)
allCounts = allWords.reduceByKey(lambda a, b: a + b)

# ("word1", count) - top 20000 words
top20000Words = allCounts.top(20000, lambda x : x[1])

# RDD that has the number 0 thru 20000
twentyK = sc.parallelize(range(20000))

# (word string, dictionary position)
topDictionary = sc.parallelize(twentyK.map(lambda x: (top20000Words[x][0], x-5)).top(19995, lambda x: x[1]))

# (word string, -1)
wordTwoVals = allCounts.mapValues(lambda x: -1).leftOuterJoin(topDictionary)

# (word string, dictionary position or -1 if not in dictionary)
totalDictionary = wordTwoVals.mapValues(lambda x: x[0] if x[1] == None else x[1])

a = totalDictionary.lookup("applicant")
print (a)

b = totalDictionary.lookup("and")
print (b)

c = totalDictionary.lookup("attack")
print (c)

d = totalDictionary.lookup("protein")
print (d)

e = totalDictionary.lookup("car")
print (e)


# This creates a vector where the value at index i is the occurences of word i in a list of word positions
def count(positions):
	a = np.zeros(19995)
	for position in positions:
		a[position] += 1
	return a

# (doc, wordCountArray)     ----     (doc, word) -> (word, doc) -> (word, (doc, position)) -> (doc, position) -> (doc, list of positions) -> (doc, count arrays)
docWordCountArrays = docAndListOfWords.flatMap(lambda x: ((x[0],j) for j in x[1])).map(lambda x: (x[1], x[0])).join(totalDictionary).map(lambda x: (x[1][0],x[1][1])).groupByKey().map(lambda x: (x[0], count(x[1])))

# Compute IDF

# number of documents in corpus
numDocs = docAndText.count()

# array [word, number of documents word is in]
wordAndNumDocs = docWordCountArrays.map(lambda x: (np.clip(x[1], 0, 1))).reduce(lambda x1, x2: np.add(x1, x2))

# IDF array
IDF = np.log(np.reciprocal(wordAndNumDocs) * numDocs)

func = np.vectorize(lambda x: 10E-10 if x == 0 else x)

IDF = func(IDF)

# Compute TF using wordCountArrays
def tf(a):
	num = np.sum(a)
	return a / num

# (doc, TF Array)
docTFArray = docWordCountArrays.map(lambda x: (x[0], tf(x[1])))

# (doc, TF-IDF array)
docAndTF_IDF = docTFArray.map(lambda x: (x[0], np.multiply(x[1], IDF)))

TF_IDFArrays = docAndTF_IDF.map(lambda x: x[1])

# Normalize Data
meanTF_IDF = TF_IDFArrays.mean()

stdTF_IDF= TF_IDFArrays.stdev()

normalizedTF_IDF = docAndTF_IDF.mapValues(lambda x: (x - meanTF_IDF) / (stdTF_IDF * 1000))

normalizedTF_IDF.cache()

# (docID, (TF_IDF vector, 1 if course case, else 0))
docXY = normalizedTF_IDF.join(docAndCourtCase)

#docXY.cache()


### LEARNING MODEL

# Loss function (inputs: trip, value from docXYTheta, d - weight on l2 term, reg - l2 term)
def LLH(trip, d, reg):
	theta = trip[2]
	y = trip[1]
	return -(theta * y) + np.log(1 + np.exp(theta)) + d*reg

# Gradient update funciton (inputs: trip - value from docXYTheta, d - weight on l2 term)
def gradFunc(trip, d):
	ar = np.zeros(19995)
	x = trip[0]
	y = trip[1]
	theta = trip[2]
	exp = np.exp(theta)
	ar = -(x * y) + (x * (exp / (1 + exp))) + 2*d*r
	# for i in range(19995):
	# 	ar[i] = -(x[i] * y) + (x[i] * (exp / (1 + exp))) + 2*d*r[i]
	return ar

# Intialize r to non-stupid guess
r = np.zeros(19995)
r = np.load("r.npy")


# Initialize change in loss function to be greater than cutoff
change = 1

# Intialize l2 term
reg = np.sum(np.power(r, 2))



# (doc ID, (TF_IDF vector, 1 if course case else 0, x•r))
docXYTheta = docXY.mapValues(lambda x: (x[0], x[1], np.dot(x[0], r)))
llh = docXYTheta.map(lambda x: LLH(x[1], d, reg)).reduce(lambda a, b: a + b)
grad = float('inf')

# Set weight on l2 term
d = 0.1

# Initialize learning rate
l = 0.00001

# Loop until change is less than cutoff
while np.abs(change) > 10E-20:
	# Compute needed values
	#docXYTheta.unpersist()
	docXYTheta = docXY.mapValues(lambda x: (x[0], x[1], np.dot(x[0], r)))
	#docXYTheta.cache()
	# Compute gradient
	prevGrad = grad
	grad = docXYTheta.map(lambda x: gradFunc(x[1], d)).reduce(lambda a, b: a + b)
	# Adjust regressors
	prevR = r
	r = prevR - l*grad
	print (r)
	# Compute new loss
	prevLLH = llh
	reg = np.sum(np.power(r, 2))
	llh = docXYTheta.map(lambda x: LLH(x[1], d, reg)).reduce(lambda a, b: a + b)
	change = prevLLH - llh
	# Bold Driver
	if llh < prevLLH:
		l *= 1.05
	else:
		l *= 0.5
	# Print LLH and
	print (llh)
	print (change)
	print ('')


# Get indeces of largest regression coefficients
top50WordIndices = np.flip(np.argsort(r)[-50:], 0)

# Get words most associated with Australian court case
topWords = totalDictionary.filter(lambda x: x[1] in top50WordIndices).map(lambda x: x[0])
print (topWords)


### EVALUATING ON TEST DATA

# Load testing data
testCorpus = sc.textFile("s3://chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt")
testValidLines = testCorpus.filter(lambda x : 'id' in x)
testDocAndText = testValidLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))
testDocAndCourtCase = testDocAndText.map(lambda x: (x[0], 1 if x[0][0:2] == 'AU' else -1))
testDocAndListOfWords = testDocAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# Compute normalized TF_IDF vecotrs on test data
testDocWordCountArrays = testDocAndListOfWords.flatMap(lambda x: ((x[0],j) for j in x[1])).map(lambda x: (x[1], x[0])).join(topDictionary).map(lambda x: (x[1][0],x[1][1])).groupByKey().map(lambda x: (x[0], count(x[1])))
testDocTFArray = testDocWordCountArrays.map(lambda x: (x[0], tf(x[1])))
testDocAndTF_IDF = testDocTFArray.map(lambda x: (x[0], np.multiply(x[1], IDF)))
testTF_IDFArrays = testDocAndTF_IDF.map(lambda x: x[1])
testNormalizedTF_IDF = testDocAndTF_IDF.mapValues(lambda x: (x - meanTF_IDF) / (stdTF_IDF * 1000))

# Get results
testResults = testNormalizedTF_IDF.mapValues(lambda x: 1 if (np.dot(x, r) > 0.0001135) else 0)

# Evaluate model
docResultAnswer = testResults.join(testDocAndCourtCase)
totalPositives = testResults.filter(lambda x: x[1] == 1).map(lambda x: x[0])
totalNegatives = testResults.filter(lambda x: x[1] == 0).map(lambda x: x[0])
falsePositives = docResultAnswer.filter(lambda x: x[1][0] == 1 and x[1][1] == 0).map(lambda x: x[0])
truePositives = docResultAnswer.filter(lambda x: x[1][0] == 1 and x[1][1] == 1).map(lambda x: x[0])
courtCases = testDocAndCourtCase.filter(lambda x: x[1] == 1).map(lambda x: x[0])
recall = (1.0 * truePositives.count()) / courtCases.count()
precision = (1.0 * truePositives.count()) / totalPositives.count()
f1 = 2 * (precision * recall) / (precision + recall)
print (recall)
print(precision)
print(f1)

# Get 3 false positives
print(sc.parallelize(falsePositives.top(3)).join(testDocAndText).map(lambda x: x[1]).collect())
