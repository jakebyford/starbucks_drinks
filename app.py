from flask import Flask, request, render_template, redirect
import pandas as pd
import pymongo
import os
# from dotenv import load_dotenv

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# MongoDB configuration
# client = pymongo.MongoClient('mongodb://localhost:27017/')
# db = client['coffee_db']
username = os.environ.get('MONGODB_USER')
password = os.environ.get('MONGODB_PASSWORD')

client = pymongo.MongoClient(f'mongodb+srv://{username}:{password}@cluster0.nrodsk8.mongodb.net/coffee_db?retryWrites=true&w=majority')
# db = client.test

# load_dotenv()
# client = MongoClient(os.environ.get('MONGO_URI'), server_api=ServerApi('1'))

db = client['coffee_db']

collection = db['preferred_drinks']
data = pd.read_csv("coffee_descriptions.csv")
data.drop_duplicates(subset=['drink_name'], inplace=True)
data = data.reset_index(drop=True)

# def getRoastLevelNumber(preferredDrink):
#     switcher = {
#         "light": 1,
#         "medium": 2,
#         "dark": 3
#     }
#     return switcher.get(preferredDrink, 0)


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
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Load the data
    data = pd.read_csv("coffee_descriptions.csv")
    data.drop_duplicates(subset=['drink_name'], inplace=True)
    data = data.reset_index(drop=True)
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
    recommendations = [i for i in (cosine_sim_df[query].sort_values(ascending=False)[1:6]).index]
    return recommendations

@app.route('/submit-form', methods=['POST'])
def submit_form():
    
    # Encode flavor notes using one-hot encoding
    # flavor_notes = request.form.getlist('flavorNotes')
    # flavor_notes_encoded = oneHotEncodeFlavorNotes(flavor_notes)
    # Create a new preferred_drink object with the encoded features
    preferred_drink = {
        'preferredDrink': request.form['drink'],
        # 'flavorNotes': flavor_notes_encoded,
        # 'brewingMethod': request.form['brewMethod'],
        # 'budget': request.form['budget']
    }
    # Save the preferred_drink object to the database
    collection.insert_one(preferred_drink)
    recommendations = coffee_similarity(preferred_drink['preferredDrink'])
    drink_names = data['drink_name']
    descriptions = data['description']
    coffeeList = [ {'drink_name': drink_names[i], 'description': descriptions[i] } for i in range(len(drink_names)) ]
    description = [i['description'] for i in coffeeList if preferred_drink['preferredDrink'] == i['drink_name']]
    return render_template('submit.html', recommendations=recommendations, preferred_drink=preferred_drink, description=description)

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
        return redirect('/thanks')
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
