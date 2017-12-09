from csv import reader
import time
import codecs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


## Collect data from csv file
def gettraining():
    csvfile = open('training.csv',newline='', encoding='utf8')
    rows = reader(csvfile,delimiter=',',quotechar='"')
    categories, text = [],[]

    for row in rows:
        categories.append(row[1])
        text.append(row[2] + ' ' + row[3])

    data = np.column_stack((text,categories))
    data = data[1:]                                         ## Exclude title row
    np.random.seed(0)
    np.random.shuffle(data)

    return data

## Collect data for unclassified products
def getunknown():
    csvfile = open('classify.csv',newline='', encoding='utf8')
    rows = reader(csvfile,delimiter=',',quotechar='"')
    text = []

    for row in rows:
        text.append(row[1] + ' ' + row[2])

    data = np.array(text[1:])
    np.random.seed(0)
    np.random.shuffle(data)

    return data

## Split data for validation of model
def splitdata(data):
    trainingfraction = 0.9
    n = data.shape[0]

    trainingdata = data[:int(trainingfraction*n)]
    validationdata = data[int(trainingfraction*n):]

    return trainingdata, validationdata

## Extract features by applying 'bag of words' and TF-IDF
def extractfeatures(X):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    X_counts = count_vect.fit_transform(X)
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    return count_vect,tfidf_transformer,X_tfidf

## Extract features by applying 'bag of words' and TF-IDF for unclassified data
def extractfeaturesunclassified(count_vect,tfidf_transformer,X):

    X_counts = count_vect.transform(X)
    X_tfidf = tfidf_transformer.transform(X_counts)

    return X_tfidf

## Train SVC classifier with text and categories
def trainclassifier(X,Y):
    clf = SVC(C=10000)
    clf.fit(X,Y)

    return clf

def main():
    start_time = time.time()

    print("Loading Data")
    trainingdata = gettraining()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Extracting features")
    count_vect, tfidf_transformer,X = extractfeatures(trainingdata[:,0])
    y = trainingdata[:,1]

    ## Create category vector
    le = LabelEncoder()
    le.fit(y)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Splitting training and validation data")
    X_train, X_validation = splitdata(X)
    TEXT_train, TEXT_validation = splitdata(trainingdata[:,0])          ## Split raw text for later analysis
    y_train, y_validation = splitdata(y)

    Y_train = le.transform(y_train)
    Y_validation = le.transform(y_validation)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Training classifier")
    clf = trainclassifier(X_train,Y_train)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Testing validation data")
    predictions  = []
    outfile = codecs.open('stackline_text_validation.txt','w','utf-8')
    for i,example in enumerate(X_validation):
        prediction = clf.predict(example)
        predictions.append(prediction)
        outfile.write("%s\t%s\t%s\n" % (TEXT_validation[i],le.inverse_transform(prediction)[0],le.inverse_transform(Y_validation[i])))

    outfile.close()

    ## This is a hack to measure the percentage of correct classifications
    validationfile = open('stackline_text_validation.txt','r')
    validationlines = validationfile.readlines()
    count = 0.
    n = len(validationlines)
    for line in validationlines:
        title,pred,label = line.strip().split('\t')
        if pred == label:
            count += 1

    percent = count/n
    print("Percent correct in validation set = %0.3f" % percent)
    print("--- %s seconds ---" % (time.time() - start_time))

    ## BEGIN UNCLASSIFIED DATA

    print("Loading unclassified data")
    classifydata = getunknown()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Extracting unclassified data features")
    X_classify = extractfeaturesunclassified(count_vect,tfidf_transformer,classifydata)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Classifying unknown data")
    classifications = []
    classifyfile = codecs.open('stackline_text_classifications.txt','w','utf-8')
    for i,example in enumerate(X_classify):
        classification = clf.predict(example)
        classifications.append(classification)
        classifyfile.write("%s\t%s\n" % (classifydata[i],le.inverse_transform(classification)[0]))

    classifyfile.close()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()