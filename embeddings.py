import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

#A script to generate LLM embeddings for the anime dataset used in the Lambda.

model = SentenceTransformer('all-MiniLM-L6-v2')

#load & inspect dataset
input_datapath = "data/anime-dataset-2023.csv"
df = pd.read_csv(input_datapath, index_col=0)
df = df.dropna()
print(len(df))
print(df.columns)

df["syn_sentence"] = df["Synopsis"].str.replace(r'[^\w\s]+', '')[:1000]

#print(df["syn_sentence"].head(3))

df["embedding"] = df["syn_sentence"].astype(str).apply(lambda x: model.encode(x))
print(df.head(2))
df.to_csv("data/anime_dataset_with_embeddings_2023.csv")


df = pd.read_csv('data/anime_dataset_with_embeddings_2023.csv')
print(df.columns)
df2 = df[df['Popularity'] > 0]
df2 = df2.sort_values(by='Popularity', ascending=True)
df3 = df2.iloc[0:10] #only take the top 1000 most popular anime!

print(df3[['Name', 'Popularity']].head(15))
print(df3[['Name', 'Popularity']].tail(15))

df3.to_csv('data/shortened_anime_dataset_with_embeddings.csv')

#+================================
# Generate similarity matrix
#+================================


# print(df.head(3))

# import ast
# import numpy as np
# from tqdm import tqdm
# from ast import literal_eval


# def process_str_to_list(s):
#     #s = "[[ 3e-1  2.12] [1.11 2.22 ] [10.0 12.0]]"
#     t = s.replace("[ ", "[")
#     st = t.replace("  ", " ")
#     ret = st.replace(" ", ",")
#     res = ast.literal_eval(ret)
#     rest = np.array(res).astype(float)
#     #rest = rest.reshape(1, -1)
#     return rest
    
# df_anime = pd.read_csv('data/anime_dataset_with_embeddings_2023.csv', converters={'COLUMN_NAME': pd.eval})
# df_anime.embedding.apply(literal_eval)

# # df_anime['embedding'] = df_anime['embedding'].apply(lambda x: process_str_to_list(x))
# # df_anime.to_csv('data/anime_embeddings_dataset.csv')


# print(df_anime['embedding'].iloc[0])
# print(type(df_anime['embedding'].iloc[0]))
# print(np.shape(df_anime['embedding'].iloc[0]))

# print('starting: ... takes 3 days!!!')
# sim_mat = []
# for i, row in tqdm(df_anime.iterrows(), total=df_anime.shape[0]):
#     a = row['embedding']
#     n = []
#     for j, row2 in df_anime.iterrows():
#         b = cosine_similarity(row2['embedding'], a)
#         n.append(b)
#     sim_mat.append(n)

# sim_mat.to_csv('data/sim_mat.csv')
