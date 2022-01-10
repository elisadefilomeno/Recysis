import pandas as pd

df = pd.read_csv("../dataset/reviews.csv")  # read original dataset
filtered_df = df[df['Review'].notnull()]    # remove rows with null review's value
filtered_df.to_csv(r'../dataset/review_notNull.csv', index=False)    # save new dataset





