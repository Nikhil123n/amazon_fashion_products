import pandas as pd
import numpy as np
import re
import string
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm

# This script contains all helper functions used for cleaning and transforming raw product data 

# These are used for text normalization, We arre using stopwords and a basic English stemmer to reduce vocab size.
# We are using Snowball stemmer, which is a bit more aggressive than Porter stemmer
# This is important for our similarity model, because we want to reduce the dimensionality of the text data 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english') 

# The weight feature is dirty, some in grams, some in kg, some in string formats (like "2kg"), and a ton of 999999999 placeholders.
# This function standardize all of that into grams. Also use eval() function to parse 2*1000 → 2000
def clean_weight(w):
    try:
        w = str(w).lower()
        w = w.replace("kg", "*1000").replace("kilograms", "*1000") 
        w = w.replace("grams", "").replace("gram", "").replace("g", "").strip()
        if "*" in w:
            w = eval(w)
        return float(w)
    except:
        return np.nan    # This will return NaN for any non-numeric or malformed weight

# This function cleans the brand features, names by lowercasing and removing punctuation 
# It also removes leading and trailing whitespace. 
def clean_brand(brand):
    return brand.lower().translate(str.maketrans('', '', string.punctuation)).strip() 

# This function removes digits and non-alphabetic characters from the product_name feature.
# It also applies stemming and removes stopword  
def remove_punctuation(text): 
    return text.translate(str.maketrans('', '', string.punctuation))

# Product names are free text but highly informative, so we need to clean them.
# We also expand contractions, remove numbers like "2pack", stem each word, and strip stopwords like "with", "the", "and"
def clean_product_name(text): 
    if pd.isna(text): 
        return ""
    text = contractions.fix(text)   # Expanding contractions like "don't" to do not   
    text = re.sub(r'\S*\d\S*', '', text)  # Removes digits and numbers
    text = re.sub('[^A-Za-z]+', ' ', text)  # Removes non-alphabetic characters
    cleaned_words = [stemmer.stem(word.lower()) for word in text.split() if word.lower() not in stop_words]
    return remove_punctuation(' '.join(cleaned_words)).strip()

# This function cleans the color feature, which is inconsistent and has multiple formats like "Red+Blue" or "Red|Blue" or "[]" 
# It splits, normalizes and deduplicates the colors to make this a usable multi-label feature.
# It also handles nulls by replacing them with unknown 
def clean_color_string(c):
    if pd.isna(c): return [] 
    c = re.sub(r'\([^)]*\)', '', c)
    c = c.replace('+', '|') 
    return list(set(filter(None, [col.strip().lower() for col in c.split('|')]))) 

# Main preprocessing function, it preprocesses the features DataFrame by applying the cleaning functions defined above.
# It also handles missing values and normalizes the data.  
# The tqdm library is used to show progress bars for the cleaning process. 
def preprocess_features(features):
    tqdm.pandas(desc="Cleaning product_name")

    # Step by step cleaning of features
    features['weight'] = features['weight'].apply(clean_weight)
    features['weight'] = features['weight'].replace(999999999, np.nan)

    features['sales_price'] = pd.to_numeric(features['sales_price'], errors='coerce')
    features['sales_price'] = features['sales_price'].fillna(features['sales_price'].median())

    features['brand'] = features['brand'].fillna('unknown')
    features['brand'] = features['brand'].apply(clean_brand)

    features['product_name'] = features['product_name'].progress_apply(clean_product_name)

    features['colour_list'] = features['colour'].apply(clean_color_string)
    features.drop(columns=['colour'], inplace=True)
    features['colour_list'] = features['colour_list'].apply(lambda x: ['unknown'] if not x else x)

    # returns the cleaned features DataFrame
    return features
