import pandas as pd


def load_data():
    # loading users
    users = pd.read_csv("./data/users.csv",
                        sep=',',
                        header=0,
                        usecols=[1, 2, 3, 4, 5],
                        index_col=0,
                        converters={2: lambda x: 0 if x == 'M' else 1,
                                    5: lambda x: int(x[:5])})

    # print(users)

    # loading movies
    movies = pd.read_csv("./data/movies.csv",
                         sep=',',
                         header=0,
                         usecols=[1, 3],
                         index_col=0,
                         converters={3: lambda x: x.split("|")},
                         quotechar="\"")
    # print(movies)

    # loading ratings
    ratings = pd.read_csv("./data/ratings.csv",
                          sep=',',
                          header=0,
                          usecols=[2, 4, 5],
                          dtype="int")

    # for index, rating in ratings.iterrows():
    #     print(list(rating))
    # print(ratings)

    return users, movies, ratings

def normalize_data(users, movies):
    # users' data
    users["age"] = users["age"] / 56
    users["occupation"] = users["occupation"] / 20
    users["zipcode"] = users["zipcode"] / 99945

    # movies' data
    genres = set(genre for genres in movies["genre"] for genre in genres)
    for row_id in range(len(movies)):
        movies.at[row_id, "genre"] = [1 if genre in movies.at[row_id, "genre"] else 0 for genre in genres]
