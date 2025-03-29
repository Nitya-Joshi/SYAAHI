import numpy as np
import os
import pandas as pd
from django.conf import settings
import ast
import nltk
import psycopg2

from django.shortcuts import render
from nltk.metrics.aline import similarity_matrix

def nav_home(request):
    return render(request, 'index.html',{'BOOKS':'http://127.0.0.1:8000/books/','MOVIES':'http://127.0.0.1:8000/movies/', 'BLEND':'http://127.0.0.1:8000/blend/'})

def login_view(request):
    return render(request, 'login.html')

def blend_view(request):
    return render(request, 'blend.html')


#BOOK Recommendation Starts here
def recommend_books():
    books_path = os.path.join(settings.BASE_DIR, 'Syaahi','books.csv')
    users_path = os.path.join(settings.BASE_DIR, 'Syaahi','users.csv')
    ratings_path = os.path.join(settings.BASE_DIR,'Syaahi','ratings.csv') 

    books = pd.read_csv(books_path, dtype={'ISBN': str}, low_memory=False)
    users = pd.read_csv(users_path)
    ratings = pd.read_csv(ratings_path)
    
    'Popularity Based Recommender System'

    ratings_with_name = ratings.merge(books,on='ISBN')
    
    # Creating the pivot table (pt)
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    eligible_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(eligible_users)]
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
    pt.fillna(0,inplace=True)

    from sklearn.metrics.pairwise import cosine_similarity
    'calculating cosine_similarity of a single row with all the other rows i.e calculating similarity of a single book with all the other books'
    similarity_scores = cosine_similarity(pt)
    
    # Create a connection to PostgreSQL
    conn = psycopg2.connect(
        dbname="book_recommendations",
        user="postgres",
        password="aaradhya13",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    similarity_matrix = []
    for i in pt.index:
        similarity = []
        for j in pt.index:
            if i!=j:
                similarity.append(float(similarity_scores(i,j)))
        
        similarity.sort()
        similarity = similarity[:2]
        sim_dict = {'isbn1': similarity[0], 'isbn2': similarity[1]}
        similarity_matrix.append(similarity[:2])       
    
    sim_mat_df = pd.DataFrame(similarity_matrix)
    sim_mat_df.to_sql('')
        
     
    
    # Insert similarity data into PostgreSQL
    for i in range(len(pt.index)):
        for j in range(i + 1, len(pt.index)):
            book1 = pt.index[i]
            book2 = pt.index[j]
            similarity_score = float(similarity_scores[i][j])  # Convert np.float64 to Python float

            # Get Book IDs or assign them programmatically
            book1_id = books[books['Book-Title'] == book1].iloc[0]['ISBN']
            book2_id = books[books['Book-Title'] == book2].iloc[0]['ISBN']

            # Insert into the 'book_similarity' table
            cursor.execute("""
                INSERT INTO book_similarity (book1_id, book2_id, similarity_score)
                VALUES (%s, %s, %s)
            """, (book1_id, book2_id, similarity_score))

    # Commit the changes and close the connection
    conn.commit()
    cursor.close()
    conn.close()
    
def recommend(book_name):
    # Fetch book ID
    conn = psycopg2.connect(dbname="your_db_name", user="postgres", password="password", host="localhost", port="5432")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT book_id FROM books WHERE title = %s
    """, (book_name,))
    book_id = cursor.fetchone()[0]

    # Query for similar books from 'book_similarity' table
    cursor.execute("""
        SELECT b2.title, bs.similarity_score 
        FROM book_similarity bs
        JOIN books b1 ON bs.book1_id = b1.id
        JOIN books b2 ON bs.book2_id = b2.id
        WHERE b1.id = %s
        ORDER BY bs.similarity_score DESC
        LIMIT 5
    """, (book_id,))
    
    similar_books = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return similar_books




def book_recommendation_view(request):
    recommended_books = recommend_books()
    return render(request, 'books.html', {'recommended_books': recommended_books})


#MOVIE Recommendation Starts here
def recommend_movies():
    movies_path = os.path.join(settings.BASE_DIR, 'Syaahi','movies.csv')
    credits_path = os.path.join(settings.BASE_DIR, 'Syaahi','credits.csv')
    
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    movies = movies.merge(credits,on='title')
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    movies.dropna(inplace=True)
    
    def convert(obj): #here obj is in string'
        L = [] 
        for i in ast.literal_eval(obj): 
            L.append(i['name'])
        return L#here object is made in a list'
    
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    
    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj): 
            if counter != 3:
                L.append(i['name'])
                counter+=1
            else:
                break
        return L
    movies['cast'] = movies['cast'].apply(convert3)
    
    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    movies['overview'] = movies['overview'].apply(lambda x:x.split()) #converts string in overview in a list
    movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
    
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id','title','tags']]
    new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
    new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(vectors)
    sorted(list(enumerate(similarity[0])),reverse=True, key=lambda x:x[1])[1:6]
    
    def recommend(movie):
        movie_index = new_df[new_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(new_df.iloc[i[0]].title)
        return recommended_movies
    return recommend('Batman Begins')

def top_movies():
    top_movies_list = ["The Godfather", "The Shawshank Redemption", "Schindler's List", 
                       "Pulp Fiction", "The Lord of the Rings", "Forrest Gump", 
                       "Star Wars", "The Dark Knight", "Fight Club", "Goodfellas"]
    return top_movies_list

def movie_recommendation_view(request):
    recommended = recommend_movies()
    top_10 = top_movies()
    return render(request, 'movies.html', {'recommended': recommend_movies, 'top_10': top_movies()})