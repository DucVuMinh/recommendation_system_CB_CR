"""

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfTransformer
from math import sqrt

class CBRS:
    def __init__(self, path_to_item_file, path_to_train, path_to_test, path_to_user_file):
        self.input_data, self.items_pre = read_item_file(path_to_item_file)

        # Create the if-idf representation for input
        self.model_tf_idf = TfidfTransformer()
        self.items_features = self.model_tf_idf.fit_transform(self.items_pre.tolist()).toarray()
        # Load matrix rating from file
        self.train = read_rating(path_to_train)
        self.test = read_rating(path_to_test)
        self.train = self.train.as_matrix()
        self.test = self.test.as_matrix()
        u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.users = pd.read_csv(path_to_user_file, sep='|', names=u_cols,encoding='latin-1')
        self.number_users = self.users.shape[0]
    def print_id(self):
        print('Content Based Recomendation System')

    def train_process(self):
        # Loop through each user and create all Coef
        number_users = self.train.shape[0]
        self.W = np.zeros(shape=(19,number_users))
        self.b = np.zeros(shape=(1, number_users))
        for i_user in range(1 , number_users):
            # Get idmovie and rate of this user
            items_id, rates = get_movie_and_rate(self.train, i_user)
            train = [self.items_features[i] for i in items_id]
            # Create a regression model and fit
            red = Ridge(alpha=0.01)
            print(type(self.items_features))
            print (type(items_id))
            red.fit(np.reshape(self.items_features[items_id,:], newshape= [-1, 19]), np.reshape(rates, newshape= [-1]))
            variables = []
            bias = []
            # Store model variables
            self.W[:,i_user -1] = red.coef_
            self.b[:,i_user - 1] = red.intercept_
            self.all_predict = self.items_features.dot(self.W) + self.b
            # Return

    def test_process(self):
        number_users = self.train.shape[0]
        # Print out a sample of user
        np.set_printoptions(precision=2) # 2 digits after .
        id_movies, rate = get_movie_and_rate(self.train, 1)
        print('Rated movies ids :', id_movies )
        print('True ratings     :', rate)
        print('Predicted ratings:', self.all_predict[id_movies, 0])

        # Print out a Train and Test Error
        print ('RMSE for training:', evaluate(self.all_predict, self.train,number_users ) )
        print ( 'RMSE for test    :', evaluate(self.all_predict, self.test, number_users) )

def evaluate(predicted, rates, n_users):
    se = 0
    cnt = 0
    for n in range(1, n_users):
        ids, scores_truth = get_movie_and_rate(rates, n)
        scores_pred = predicted[ids, n-1]
        e = scores_truth - scores_pred
        e = np.reshape(e, newshape=(-1,1))
        se += (e*e).sum(axis = 0)
        cnt += e.size
    return sqrt(se/cnt)

def read_item_file(path_to_item_file):
    # Load input from file
    col_names = ['movie id', 'movie title' , 'realese date', 'video release date', 'IMDb Url',
                 'unknown', 'Animation', 'Children', ' Comedy', 'Crime', 'Documentary',
                 'Drama', 'Fantasy', 'Film-noir', 'Honnor', 'Musical', 'Mystery',
                 'Romantic', 'Sic-Fi', 'Thriller', 'War' , 'Western']
    input_data = pd.read_csv(path_to_item_file, sep='|', encoding = 'latin-1',
                             names = col_names)
    matrix = input_data.as_matrix()

    # Create Items content
    items_pre = matrix[:,-19:]
    return input_data, items_pre


def read_rating(path_to_file):
    input_data = pd.read_csv(path_to_file, sep='\t',
                             encoding = 'latin-1',
                             names= ['User id', 'Movie id', 'Rate', 'Time Stamp'])
    return input_data

def get_movie_and_rate(rate_matrix,id_user):
    # Get all id of users
    id_users = rate_matrix[:,0]
    # Get items rated from user
    items = np.where(id_users == id_user)
    items_id = rate_matrix[items , 1]
    rate = rate_matrix[items, 2]
    return items_id, rate


if __name__ == '__main__':
    cbrs = CBRS('../data/ml-100k/u.item', '../data/ml-100k/ua.base', '../data/ml-100k/ua.test',
                '../data/ml-100k/u.user')
    cbrs.train_process()
    cbrs.test_process()

