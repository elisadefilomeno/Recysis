import dill as pickle
import pandas as pd


comments_df=pd.read_csv("../dataset/modified_comments.csv")
count_vect_model = pickle.load(open('../models/count_vect.sav', 'rb'))
tfidf_model = pickle.load(open('../models/tfidf_model.sav', 'rb'))
loaded_model = pickle.load(open("../models/static_model.sav", 'rb'))

X_test = comments_df['Review'].values


def static_model(loaded_model, X_test):

    X_test_counts = count_vect_model.transform(X_test)
    X_test_tfidf = tfidf_model.transform(X_test_counts)

    predicted = loaded_model.predict(X_test_tfidf)
    comments_df['Rating']=predicted
    comments_df.to_csv(r'../dataset/predicted_all_comments.csv', index=False)


if __name__ == '__main__':

    static_model(loaded_model, X_test)