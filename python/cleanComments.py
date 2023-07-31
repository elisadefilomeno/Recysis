import pandas as pd


def modifyStr(x):
    x = x.replace('\r', '')
    x = x.replace('\n', '')
    x = x.replace('  ', '')
    x = x.replace('&', 'and')
    return x


df = pd.read_csv("../dataset/review_notNull.csv")  # read original dataset
df['Review'] = df['Review'].apply(modifyStr)
df.to_csv(r'../dataset/modified_comments.csv', index=False)    # save new dataset
