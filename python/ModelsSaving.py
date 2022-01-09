import pandas as pd
import dill as pickle
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import EnglishStemmer


df=pd.read_csv("../dataset/dataset_2008.csv")

classifier = svm.LinearSVC(C=0.1)
X_train = df['Review'].values
y_train = df['Rating'].values


if __name__ == '__main__':
    stemmer = EnglishStemmer()
    analyzer = CountVectorizer().build_analyzer()
    count_vect = CountVectorizer(stop_words='english', strip_accents='ascii', lowercase=True,
                                 token_pattern=r"(?u)\b[^\d\W][^\d\W]+\b", analyzer=lambda doc:(stemmer.stem(w) for w in analyzer(doc)))
    tfidf_transformer = TfidfTransformer()

    # Text preprocessing, tokenizing and filtering of stopwords
    X_train_counts = count_vect.fit_transform(X_train)
    pickle.dump(count_vect, open('../models/count_vect.sav', 'wb'))



    # From occurrences to frequencies
    X_train_tfidf = tfidf_transformer.fit(X_train_counts)
    pickle.dump(X_train_tfidf, open('../models/tfidf_model.sav', 'wb'))
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Training classifier with initial training set
    classifier.fit(X_train_tfidf, y_train)
    pickle.dump(classifier, open('../models/static_model.sav', 'wb'))





