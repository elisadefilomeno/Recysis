import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import EnglishStemmer
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay


df=pd.read_csv("../dataset/dataset_2008.csv")
df1=pd.read_csv("../dataset/evento1.csv")
df2=pd.read_csv("../dataset/evento2.csv")
df3=pd.read_csv("../dataset/evento3.csv")
df4=pd.read_csv("../dataset/evento4.csv")
df5=pd.read_csv("../dataset/evento5.csv")
df6=pd.read_csv("../dataset/evento6.csv")


# global variables
classifier =svm.LinearSVC(C=0.1)
stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemming(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

count_vect = CountVectorizer(stop_words='english', strip_accents='ascii', lowercase=True,
                             token_pattern=r"(?u)\b[^\d\W][^\d\W]+\b", analyzer=stemming)
tfidf_transformer = TfidfTransformer()
X_initial_train = df['Review'].values
y_initial_train = df['Rating'].values

# Text preprocessing, tokenizing and filtering of stopwords
X_initial_train_counts = count_vect.fit_transform(X_initial_train)

# From occurrences to frequencies
X_initial_train_tfidf = tfidf_transformer.fit_transform(X_initial_train_counts)

sliding_window_accuracies = []
sliding_dictionary_size=[]


# Training classifier with initial training set
classifier.fit(X_initial_train_tfidf, y_initial_train)

number_event=1
X_train = X_initial_train
y_train = y_initial_train

while number_event<=6:
    current_event_dataset = globals()["df" + str(number_event)]

    # building the training set with 60 new comments from the previous event
    # and removing 60 oldest comments
    if number_event>1:
        print("number event: ", number_event)
        previous_event_dataset_string = "df"+str(number_event-1)
        previous_event_dataset = globals()[previous_event_dataset_string]
        print("dataset iniziale", X_train.shape)
        X_train = np.append(X_train, previous_event_dataset['Review'].values)
        y_train = np.append(y_train, previous_event_dataset['Rating'].values)

        X_train = X_train[60:]
        y_train = y_train[60:]

    # Text preprocessing, tokenizing and filtering of stopwords
    X_train_counts = count_vect.fit_transform(X_train)
    # From occurrences to frequencies
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    classifier.fit(X_train_tfidf, y_train)

    X_test = current_event_dataset['Review'].values
    y_test = current_event_dataset['Rating'].values

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = classifier.predict(X_test_tfidf)
    sliding_dictionary_size.append(len(count_vect.vocabulary_))
    sliding_window_accuracies.append(accuracy_score(y_test, predicted))

    print(metrics.classification_report(y_test, predicted))
    ConfusionMatrixDisplay.from_predictions(y_test,predicted)
    plt.title("SLIDING MODEL")
    plt.show()
    number_event=number_event+1


print("sliding window accuracies: ", sliding_window_accuracies)
print("sliding window dictionary size:", sliding_dictionary_size)

