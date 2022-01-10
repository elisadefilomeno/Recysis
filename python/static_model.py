import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
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

def stemming(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def static_model(classifier,number_event):
    event_dataset=globals()["df" + str(number_event)]
    X_test = event_dataset['Review'].values
    y_test = event_dataset['Rating'].values

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = classifier.predict(X_test_tfidf)
    print(metrics.classification_report(y_test, predicted))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    plt.title("STATIC MODEL")
    plt.show()
    return accuracy_score(y_test, predicted)


if __name__ == '__main__':
    classifier = svm.LinearSVC(C=0.1)
    stemmer = EnglishStemmer()
    analyzer = CountVectorizer().build_analyzer()
    count_vect = CountVectorizer(stop_words='english', strip_accents='ascii', lowercase=True,
                                 token_pattern=r"(?u)\b[^\d\W][^\d\W]+\b", analyzer=stemming)
    tfidf_transformer = TfidfTransformer()

    X_initial_train = df['Review'].values
    y_initial_train = df['Rating'].values
    static_accuracies = []
    incremental_accuracies = []
    static_dictionary_size = []

    # Text preprocessing, tokenizing and filtering of stopwords
    X_initial_train_counts = count_vect.fit_transform(X_initial_train)

    # From occurrences to frequencies
    X_initial_train_tfidf = tfidf_transformer.fit_transform(X_initial_train_counts)

    for i in range(6):
        # Training classifier with initial training set
        classifier.fit(X_initial_train_tfidf, y_initial_train)
        static_dictionary_size.append(len(count_vect.vocabulary_))
        print("event ", i + 1)

        string_dataset = "df" + str(i + 1)
        event_dataset = globals()[string_dataset]
        static_accuracies.append(static_model(classifier,i+1))
    print("static accuracies: ", static_accuracies)
    print("static dictionary size: ", static_dictionary_size)




