# Anime Recommendation end to end deployment


### NCF
__What is NCF?__
* Collaborative filtering, a very commonly used method in recommendation systems, essentially decomposes a matrix of users and their ratings or interactions with items into the product of two matrices. By doing so, one can obtain embeddings of items and users with respect to some latent feature dimension. This allows for an understanding of users and items in the context of this latent dimension, and extrapolation of similarity for items or users that have not yet interacted.
![Example of matrix decomposition.](https://www.researchgate.net/profile/Andrei-Chupakhin/publication/343333900/figure/fig1/AS:919202034626560@1596166251035/Example-of-matrix-decomposition.ppm)
* Item user interactions in this latent space can be more complex than just the dot product operation used to concatenate the embeddings. *Neural collaborative filtering* adds a feedforward neural network on top of the embeddings to represent a more complex, nonlinear function.
* The following neural collaborative filtering model was implemented, following guidance of the original paper:
![Neural Collaborative Filtering. Supercharging collaborative filtering… | by  Abhishek Sharma | Towards Data Science](https://miro.medium.com/v2/resize:fit:1400/1*aP-Mx266ExwoWZPSdHtYpA.png)
Code for the neural network implementation can be found here: KAGGLE NOTEBOOK
(some inspiration from https://www.kaggle.com/shivakumarasam)

### Serverless Orchestration

![image](https://github.com/ubitquitin/mal_reccos/assets/14205051/39cf7566-b44e-42be-bab8-74e5c6c44a30)

In order to create a machine learning powered anime recommendation website, the following infrastructure was implemented using the [serverless framework](https://www.serverless.com/). This framework allows for easy deployment of AWS lambda services with infrastructure as code style deployment. 

To hold sklearn, numpy and pandas libraries, a docker image was created and pushed to ECR. The Lambda was configured to run on this container to allow for encoding of incoming anime queries to the embedding space of the NCF model. From there, pandas and numpy functions were used to calculate the top n most similar anime in the dataset.

Because the pandas dataset load was expensive, caching was used to make multiple requests after the Lambda warmed up much faster. 

### LLM

As an extra exploration step, HuggingFace's [sentence-transformer](https://www.sbert.net/) library was used to encode anime synopses as large language model numerical vectors. This allowed for a similarity score to be computed pairwise between all of the anime in the dataset, that would be based on a language model's understanding of the text similarity. So instead of user based ratings implemented in the NCF, the similarity of the text of the synopsis was used as an alternative recommendation style. 

### Similarity score

Dot product was used for NCF similarity score.
Cosine similarity was used for LLM similarity.
Cosine similarity is simply dot product normalized by the scalar product of two matrices: ![Cosine similarity formula | Download Scientific Diagram](data:image/png;base64,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)
* This is a desirable metric for word vector embeddings if we do not want to consider the magnitude, or occurrence of a word. 
* Dot product was used for the NCF approach because it results in a faster calculation. Because the user-item dataset fed to the NCF model was just binary in data (1 = interaction, 0 = no interaction), the resulting embeddings were already normalized.
