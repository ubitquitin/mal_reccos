import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import boto3
from io import BytesIO

import logging
log = logging.getLogger()
log.setLevel(logging.INFO)

# def main(event, context):

#     input = json.loads(event['body'])
#     n = int(input["num"])
#     arr = np.array(np.arange(n)).astype(np.float64)

#     # Convert the NumPy array to a list
#     arr_as_list = arr.tolist()

#     # Serialize the list
#     response = {"statusCode": 200, "body": json.dumps(arr_as_list)}

#     return response


def lambda_handler(event, context):
    input = json.loads(event['body'])
    n = input['n']
    name = input['name']
    
    log.info('entering function:')
    topn = find_similar_animes(name, n)
    
    return {
        'statusCode': 200,
        'body': json.dumps(topn)
    }

#inputs: df_anime, item_enc, anime_weights
def find_similar_animes(name, n=10, return_dist=False, neg=False):
    
    pd.set_option('display.max_colwidth', None)
    s3 = boto3.client('s3')
    
    #anime dataframe
    key = 'anime-dataset-2023.csv'
    bucket = 'mlopsproj4'
    response = s3.get_object(Bucket=bucket, Key=key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 200:
        df_anime = pd.read_csv(response.get("Body"))
        log.info('dataframe read.')
    
    #NCF anime embeddings
    key = 'anime_weights.npy'
    bucket = 'mlopsproj4'
    response = s3.get_object(Bucket=bucket, Key=key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 200:
        anime_weights = np.load(BytesIO(response.get("Body").read()))
        log.info('anime embeddings recieved.')
    
    #Anime label encoder from training
    key = 'classes.npy'
    bucket = 'mlopsproj4'
    item_enc = LabelEncoder()
    response = s3.get_object(Bucket=bucket, Key=key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 200:
        item_enc.classes_ = np.load(BytesIO(response.get("Body").read()))
        log.info('item encoder weights loaded')
    
    try:
        anime_row = df_anime[df_anime['Name'] == name].iloc[0]
        index = anime_row['anime_id']
        log.info('encoding anime input.')
        encoded_index = item_enc.transform([index])[0]
        weights = anime_weights
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        log.info('sorting distances of anime.')
        n = n + 1            
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]
        log.info('Animes closest to {}'.format(name))
        if return_dist:
            return dists, closest
        
        SimilarityArr = []
        
        log.info('Getting specific anime info...')
        for close in closest:
            decoded_id = item_enc.inverse_transform([close])[0]
            anime_frame = df_anime[df_anime['anime_id'] == decoded_id]
            
            anime_name = anime_frame['Name'].values[0]
            english_name = anime_frame['English name'].values[0]
            name = english_name if english_name != "UNKNOWN" else anime_name
            genre = anime_frame['Genres'].values[0]
            Synopsis = anime_frame['Synopsis'].values[0]
            similarity = dists[close]
            similarity = "{:.2f}%".format(similarity * 100)
            SimilarityArr.append({"Name": name, "Similarity": similarity, "Genres": genre, "Synopsis":Synopsis})
        Frame = pd.DataFrame(SimilarityArr).sort_values(by="Similarity", ascending=False)
        #print(Frame[Frame.Name != name].head(n))
        ret_df = Frame[Frame.Name != name]
        resp = ret_df.to_json()
        return resp
    except:
        print('{} not found in Anime list'.format(name))
        
        
        
if __name__ == "__main__":
    lambda_handler('', '')