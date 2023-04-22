from flask import Flask, request, render_template, redirect
from pymongo import MongoClient
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
# from dotenv import load_dotenv

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# MongoDB configuration for localhost
# client = pymongo.MongoClient('mongodb://localhost:27017/') 
# db = client['coffee_db']


# Set up Config var in Heroku under app Settings. Here's an example:
# MONGO_URI = mongodb+srv://{username}:{password}@cluster0.nrodsk8.mongodb.net/{database_name}?retryWrites=true&w=majority'
#
# Here we are calling the Config variable we created in  'MONGO_URI'
mongo_uri = os.environ.get('MONGO_URI')


# load_dotenv()
client = MongoClient(mongo_uri)

db = client['coffee_db']

collection = db['preferred_drinks']
data = pd.read_csv("coffee_descriptions.csv")
data.drop_duplicates(subset=['drink_name'], inplace=True)
data = data.reset_index(drop=True)

# get database and collections(tables)
user = str(os.environ.get('USER'))
password = str(os.environ.get('PASS'))
cluster = MongoClient(f"mongodb+srv://{user}:{password}@cluster0.nrodsk8.mongodb.net/coffee_db?retryWrites=true&w=majority")
db = cluster['coffee_db']
surveys = db['surveys']
surveys = list(surveys.find())
surveys = pd.DataFrame(surveys)

clean_drink_names = [i for i in data['drink_name']]

users = [i for i in surveys['_id']]

# clean the users
clean_users = []
count = 0
for i in users:
    new_id = "user" + str(count)
    count += 1
    clean_users.append(new_id)

ratings = [i for i in surveys['ratings']]

clean_ratings = []
for i in ratings:
    clean_i = []
    for j in i:
        if j == 'N/A':
            j = np.nan
        if j == '1':
            j = 1
        if j == '2':
            j = 2
        if j == '3':
            j = 3
        if j == '4':
            j = 4
        if j == '5':
            j = 5
        clean_i.append(j)
    clean_ratings.append(clean_i)

clean_ratings_df = pd.DataFrame(clean_ratings, index=clean_users, columns=clean_drink_names)

clean_ratings_cosine_similarity = cosine_similarity(clean_ratings_df.fillna(0))

# Create a dataframe with the cosine similarity matrix
clean_ratings_cosine_similarity_df = pd.DataFrame(clean_ratings_cosine_similarity, columns=clean_users, index=clean_users)

# def oneHotEncodeFlavorNotes(flavorNotes):
#     # Array with length equal to the number of flavor notes you want to encode
#     encoding = [0, 0, 0]
#     for note in flavorNotes:
#         if note == 'fruity':
#             encoding[0] = 1
#         elif note == 'chocolaty':
#             encoding[1] = 1
#         elif note == 'nutty':
#             encoding[2] = 1
#     return encoding



@app.route('/', methods=['GET'])
# def index():
#     coffeeList = dropdown()
#     return render_template('index.html', coffeeList=coffeeList)

def index(data=data):
    return render_template('index.html', data=data)

def coffee_similarity(preferredDrink):
    # Preprocess the text
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['description'])

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(X)

    # Create a dataframe with the cosine similarity matrix
    cosine_sim_df = pd.DataFrame(cosine_sim, columns=data['drink_name'], index=data['drink_name'])

    query = preferredDrink
    # Print the cosine similarity matrix
    cosine_sim_df = cosine_sim_df.sort_values(by=[query], ascending=False)
    query = preferredDrink
    recommendations = [i for i in (cosine_sim_df[query].sort_values(ascending=False)[1:4]).index]
    return recommendations
    

def generate_recommendations(top_n=5):

    target_user = clean_ratings_df[-1:].index[0]
    
    # Extract the similarity scores for the target user from the cosine similarity matrix
    similarity_scores = clean_ratings_cosine_similarity_df[target_user]

    # Sort the similarity scores in descending order and select the top N similar users
    similar_users_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]

    # Get the ratings of similar users for items that the target user has not rated
    similar_users_ratings = clean_ratings_df.iloc[similar_users_indices]
    target_user_ratings = clean_ratings_df.loc[target_user]

    # Filter out items that the target user has already rated
    unrated_items_mask = target_user_ratings.isna()
    similar_users_ratings = similar_users_ratings.loc[:, unrated_items_mask]

    # Calculate the weighted average of the ratings of similar users using the similarity scores as weights
    similarity_scores = similarity_scores[similar_users_indices]
    weighted_avg_ratings = similar_users_ratings.mul(similarity_scores, axis=0).sum() / similarity_scores.sum()

    # Sort the weighted average ratings in descending order to get the top-rated items
    recommended_items = weighted_avg_ratings.sort_values(ascending=False)

    # You can now access the recommended items for the target user in the `recommended_items` Series.
    # For example, to get the top N recommended items for the target user:
    top_n_recommended_items = recommended_items.head(3)
    # print(top_n_recommended_items)

    ###### EXTRA CODE for MAE or other evaluation metric ##########
    # Extract the actual ratings for the recommended items
    # actual_ratings = clean_ratings_df.fillna(0).loc[f"user{target_user}", top_n_recommended_items.index]

    # Extract the predicted ratings (weighted average ratings) for the recommended items
    # predicted_ratings = top_n_recommended_items.fillna(0).values


    return top_n_recommended_items.keys()

@app.route('/submit-form', methods=['POST'])
def submit_form():
    
    # Encode flavor notes using one-hot encoding
    # flavor_notes = request.form.getlist('flavorNotes')
    # flavor_notes_encoded = oneHotEncodeFlavorNotes(flavor_notes)

    # Create a new preferred_drink object with the encoded features
    preferred_drink = {
        'preferredDrink': request.form['drink']
        # 'flavorNotes': flavor_notes_encoded,
        # 'brewingMethod': request.form['brewMethod'],
        # 'budget': request.form['budget']
    }
    # Save the preferred_drink object to the database
    db.preferred_drinks.insert_one(preferred_drink)
    item_recommendations = coffee_similarity(preferred_drink['preferredDrink'])
    user_recommendations = generate_recommendations(top_n=5)
    drink_names = data['drink_name']
    descriptions = data['description']
    coffeeList = [ {'drink_name': drink_names[i], 'description': descriptions[i] } for i in range(len(drink_names)) ]
    description = [i['description'] for i in coffeeList if preferred_drink['preferredDrink'] == i['drink_name']]
    return render_template('submit.html', user_recommendations=user_recommendations, item_recommendations=item_recommendations, preferred_drink=preferred_drink, description=description)

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    coffees = enumerate(data['drink_name'])
    if request.method == 'POST':
        ratings = []
        drinks_tried = []
        for index, items in coffees:
            rating = request.form.get(f'rating{index+1}')
            ratings.append(rating)

            answer = request.form.get(f'drink{index+1}')
            drinks_tried.append(answer)
        responses = {
            "drinks_tried": drinks_tried,
            "ratings": ratings
        }

        db.surveys.insert_one(responses)
        return redirect('/')
        # return 'Thanks for your response!'
    return render_template('survey.html', coffees=coffees)

# @app.route('/success', methods=['POST'])
# def success():
#     return redirect(url_for('thanks'))

@app.route('/thanks')
def thanks():
    return 'Thanks for submitting the form!'

# def get_roast_level_number(preferredDrink):
#     switcher = {
#         'light': 1,
#         'medium': 2,
#         'dark': 3
#     }
#     return switcher.get(preferredDrink, 0)


if __name__ == "__main__":
    app.run(debug=True)
