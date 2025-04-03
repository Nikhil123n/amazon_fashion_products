import pandas as pd
import numpy as np
import os
import re
import math
from typing import List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_weight, clean_brand, clean_product_name, clean_color_string

# Global variables to store processed data
features = None
feature_matrix = None
uniq_ids = None


def prepare_system():  
    global features, feature_matrix, uniq_ids

    # Load the LDJSON file containing the 30K fashion product records  
    # These are scraped samples, and we're only keeping the core features useful for similarity  
    file_path = os.path.join("data", "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson")
    amz_data = pd.read_json(file_path, lines=True) 
    features = amz_data[['uniq_id', 'brand', 'sales_price', 'weight', 'rating', 'colour', 'product_name']].copy() 

    ### Preprocessing
    # Cleaning the weight feature first. It's very noisy: 
    # Different units like kg, g, grams, 999999999 placeholder, and even malformed strings are present in this weight feature 
    # We are converting everything to grams and parsing expressions like 2Kg to 2000 
    # This is a critical feature, so we impute, because it's having placeholders in almost 48% of rows but still important 
    features['weight'] = features['weight'].apply(clean_weight) 
    features['weight'] = features['weight'].replace(999999999, np.nan) 

    # Cleaning the sales_price feature, because it's a key factor users care about alot. . 
    # Since dtype of this feature is object, while it should be float - we need to convert it to numeric 
    # We force non-numeric entries like "-" to NaN and impute with the median to be robust against outliers. 
    features['sales_price'] = pd.to_numeric(features['sales_price'], errors='coerce')
    features['sales_price'].fillna(features['sales_price'].median(), inplace=True) 

    # Standardizing brand names by filling nulls with 'unknown' and lowercasing, removing punctuations, whitespaces. 
    # This avoids splitting identical brands written differently. 
    features['brand'] = features['brand'].fillna('unknown').apply(clean_brand) 
    
    # product_name is a free-text column but highly descriptive of the item .
    # We chose this feature, because this will add more semantic understanding of the product to our similarity model. 
    # We clean it using stemming, remove stopwords, and normalize punctuation/digits.  
    features['product_name'] = features['product_name'].fillna('').apply(clean_product_name) 

    # colour feature is inconsistent and some rows have multiple colors like "Red+Blue" or "Red|Blue" or "[]"
    # We're splitting, normalizing and deduplicating to make this a usable multi-label feature, also it has around 23971 null rows, need to be taken care.
    features['colour_list'] = features['colour'].apply(clean_color_string) 
    features.drop(columns=['colour'], inplace=True) 
    features['colour_list'] = features['colour_list'].apply(lambda x: ['unknown'] if not x else x) 

    ### Imputate missing weights of weight feature using Random Forest 
    # Now we impute the missing weights using a Random Forest regressor. 
    # Reason for RandomForestRegressor: because it handles non-linear relationships better and avoids overfitting compared to decision trees regressor
    # After experimenting with multiple models, Random Forest Regressor - performed the best. 
    # We include product_name via. TF-IDF, to capture text semantics and brand/rating/price to improve the prediction.
    df = features[['brand', 'product_name', 'sales_price', 'rating', 'weight']].copy() 
    df['brand_encoded'] = LabelEncoder().fit_transform(df['brand']) 

    #  TF-IDF for product_name helps us capture textual features in a low-dimensional space
    tfidf = TfidfVectorizer(max_features=100)    
    text_df = pd.DataFrame(tfidf.fit_transform(df['product_name']).toarray(), columns=tfidf.get_feature_names_out())  
    # Creating final feature matrix which combines all the selected features of our interest, for training our weight pr edictor.
    X_full = pd.concat([df[['brand_encoded', 'sales_price', 'rating']].reset_index(drop=True), text_df], axis=1)


    # We need to drop the rows with NaN in the target variable (weight) for training.    
    # We also need to drop the rows with NaN in the features (product_name, brand, sales_price, rating) for training.        
    known_mask = ~df['weight'].isna()           
    X_eval = X_full[known_mask]
    y_eval = df.loc[known_mask, 'weight']         

    # Remove unrealistic outliers like weights > 10kg (likely data entry issues for the amazon fashion dataset 
    # We use a threshold of 5000 grams (5kg) to filter out unrealistic weights.
    # This is a based on domain knowledge and can be adjusted as needed.
    mask = y_eval < 5000
    X_eval = X_eval[mask]       
    y_eval = y_eval[mask]


    # Random Forest works well here due to its robustness with high-cardinality categorical data , like brands.
    # We use max_features='log2' to reduce the number of features considered for each split, which helps in reducing overfitting.
    # We also set random_state for reproducibility.
    forest = RandomForestRegressor(n_estimators=100, max_depth=None, max_features='log2', random_state=42)
    forest.fit(X_eval, y_eval)     


    # Now we predict weights for missing rows in the original dataset.    
    # We use the same feature matrix as before, but now we need to predict the missing weights.
    X_missing = X_full[~known_mask]          
    predicted_missing_weights = forest.predict(X_missing)       
    features.loc[~known_mask, 'weight'] = predicted_missing_weights
    features = features[features['weight'] <= 4000] # Again, filter out unrealistic values for downstream similarity calculations (weights > 4kg)

    ### Final cleanup
    # Remove duplicate products to avoid noise in similarity search.   
    features.drop_duplicates(subset=['product_name'], inplace=True)
    # Remove rows with NaN in the following key features
    features.dropna(subset=['brand', 'sales_price', 'weight', 'rating'], inplace=True) 
    features.reset_index(drop=True, inplace=True)   

    ### Feature matrix creation
    # We create a feature matrix that combines all the features we want to use for similarity calculations.  
    # - Brand = OneHot for categorical identity
    # - Sales/Weight/Rating = MinMax normalized to keep scale uniform
    # - Product Name = TF-IDF to capture semantic similarity
    # - Colour = MultiLabelBinarizer since each product can have multiple values
    product_name_tfidf = TfidfVectorizer(max_features=100).fit_transform(features['product_name'])
    color_matrix = MultiLabelBinarizer().fit_transform(features['colour_list'])
    brand_encoded = OneHotEncoder(handle_unknown='ignore').fit_transform(features[['brand']])
    numeric_features = MinMaxScaler().fit_transform(features[['sales_price', 'weight', 'rating']])


    # We concatenate all the features into a single feature matrix for similarity calculations.
    feature_matrix = np.hstack([
        brand_encoded.toarray(),
        numeric_features,
        product_name_tfidf.toarray(),
        color_matrix
    ])


    uniq_ids = features['uniq_id'].tolist()


def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    # Classic cosine similarity returns num_similar most alike - products based on our final matrix
    # We use cosine similarity to find the most similar products based on the feature matrix.
    # The function takes a product ID and the number of similar products to return.
    # It raises a ValueError if the product ID is not found in the cleaned dataset. 
    global features, feature_matrix, uniq_ids

    if product_id not in uniq_ids:
        raise ValueError(f"Product ID {product_id} not found in cleaned dataset.")

    idx = uniq_ids.index(product_id)
    query_vector = feature_matrix[idx].reshape(1, -1) # Reshape to 2D for cosine similarity calculation
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()   # Flatten to 1D array
    # Get the indices of the most similar products, excluding the query product itself.
    # Get top-N indices sorted by similarity
    top_indices = np.argsort(-similarities)  # already descending

    # Skip the query product itself
    top_indices = [i for i in top_indices if i != idx]
    # Get similarity and tiebreaker values
    top_data = [
        (i, similarities[i], features.iloc[i]['sales_price'])  # or 'ratings'
        for i in top_indices
    ]

    # Now sort: first by similarity, then by higher rating
    top_data_sorted = sorted(top_data, key=lambda x: (-x[1], -x[2]))
    # Get the final top-N ids
    valid_top_indices = [i for i, _, _ in top_data_sorted[:num_similar]]


    similar_ids = [uniq_ids[i] for i in valid_top_indices]
    return similar_ids
