import pandas as pd

# Import data using datetime and no data value
df_input = pd.read_csv("../dataset/modified_comments.csv", usecols=['DateModified','Review', 'Rating'])

df_output=df_input.loc[df_input['Rating'].isin([1,3,5])]

df_output['Rating']=df_output['Rating'].replace(1,'negative')
df_output['Rating']=df_output['Rating'].replace(3,'neutral')
df_output['Rating']=df_output['Rating'].replace(5,'positive')

print("negative:", df_output.Rating.value_counts()['negative'])
print("positive:",df_output.Rating.value_counts()['positive'])
print("neutral:",df_output.Rating.value_counts()['neutral'])

dataset_negative = df_output.loc[df_output['Rating']=='negative']
dataset_negative = dataset_negative[:1500]

dataset_positive = df_output.loc[df_output['Rating']=='positive']
dataset_positive = dataset_positive[:1500]

dataset_neutral = df_output.loc[df_output['Rating']=='neutral']
dataset_neutral = dataset_neutral[:1500]

df_output=pd.merge(dataset_negative, dataset_positive, how='outer')
df_output=pd.merge(df_output, dataset_neutral, how='outer')

df_output = df_output.sort_values(['DateModified'])

df_output=df_output[['Review', 'Rating']]

df_output.to_csv(r'../dataset/dataset_2008.csv', index=False)
