import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load data
interactions_df = pd.read_csv('dataset/RAW_interactions.csv')
recipes_df = pd.read_csv('dataset/RAW_recipes.csv')

# Rename columns for clarity
recipes_df.columns = ['recipe_name', 'recipe_id', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients']

# Convert necessary columns to integers
recipes_df['recipe_id'] = recipes_df['recipe_id'].astype(int)

# Handle missing values
interactions_df = interactions_df.dropna()
recipes_df = recipes_df.dropna()

# Convert ID columns to integers
interactions_df['user_id'] = interactions_df['user_id'].astype(int)
interactions_df['recipe_id'] = interactions_df['recipe_id'].astype(int)

# Prepare data for surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions_df[['user_id', 'recipe_id', 'rating']], reader)

# Split data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD algorithm for collaborative filtering
algo = SVD()

# Train the algorithm on the training set
algo.fit(trainset)

# Predict ratings for the test set
predictions = algo.test(testset)

# Function to get top N recommendations for a user
def get_top_n_recommendations(user_id, n=10):
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    user_rated_recipes = user_interactions['recipe_id'].tolist()

    all_recipes = recipes_df['recipe_id'].tolist()
    unrated_recipes = [recipe for recipe in all_recipes if recipe not in user_rated_recipes]

    predicted_ratings = []
    for recipe_id in unrated_recipes:
        predicted_rating = algo.predict(user_id, recipe_id).est
        predicted_ratings.append((recipe_id, predicted_rating))

    top_n_recommendations = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:n]

    return [recipe_id for recipe_id, rating in top_n_recommendations]

# Example usage
user_id = 1  # Replace with a valid user ID from the interactions data
top_n = get_top_n_recommendations(user_id, n=10)
print("Top N Recommendations for User {}: {}".format(user_id, top_n))
