
import pandas as pd
import matplotlib.pyplot as plt

from datetime import timedelta 

class SampleData:
    def __init__(self, data_csv_str):
        self.data_df = pd.read_csv(data_csv_str)

        self.data_df = self.data_df.drop(['video_id', 'title', 'channel_title', 'tags',
                      'thumbnail_link', 'comments_disabled', 'ratings_disabled',
                      'video_error_or_removed',
                      'description'], axis=1)

        #import trending date and add 1 day to it, so the publish date becomes always before the trending date.
        trending_dates = pd.to_datetime("20"+ self.data_df['trending_date'] , format="%Y.%d.%m") + timedelta(days=1)

        #import publish_date
        publish_times = pd.to_datetime(self.data_df['publish_time'] , infer_datetime_format = True)

        #Take the difference between the publishing date and trending date "how long does the video take to be in trending"
        trending_time = trending_dates.subtract(publish_times)
        # add a column trending time (in second)
        self.data_df['trending_time'] =  trending_time.dt.total_seconds()
        #Drop publish time because we will not use it. We will use trending_time instead
        self.data_df = self.data_df.drop(['publish_time'], axis=1)

        # List of unique trending date
        list_date = self.data_df.trending_date.unique().tolist()
        #List of dataframes. Each data frame contains one trending date.

        self.df_by_trending_date = []
        for i in list_date:
            self.df_by_trending_date.append(self.data_df.loc[self.data_df.trending_date==i])

    def get_total_df(self):
        return self.data_df

    def get_publish_date_df_list(self):
        return self.df_by_trending_date

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
class PcaDecomp:
    def __init__(self, df, features_str_list):
        features_str_list = features_str_list

        # Separating out the features
        X = df.loc[:, features_str_list].values
        # Separating out the target

        #PCA
        # Standardizing the features
        X = StandardScaler().fit_transform(X)
        self.pca = PCA()#n_components=4)
        self.pca.fit_transform(X)
        transformed_data = self.pca.fit_transform(X)

        self.decomposed_df = pd.DataFrame(data = transformed_data,
                                   columns = self.gen_pca_name_list(len(features_str_list)))

    def get_explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_

    def get_pca_df(self):
        return self.decomposed_df

    def gen_pca_name_list(self, num_features):
        print(num_features)
        component_name_list = []
        for i in range(1,num_features+1):
            component_name_list.append('pc'+str(i))
        print(component_name_list)
        return component_name_list

from sklearn.model_selection import train_test_split
if __name__ == '__main__':

    sample = SampleData(r'USvideos.csv')
    df = sample.get_total_df()
    X = df.loc[:, ['views', 'comment_count', 'dislikes']]
    y = df.loc[:, ['likes']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 69)

    from sklearn.ensemble import RandomForestRegressor
    forest = RandomForestRegressor(max_depth = 5)
    forest.fit(X_train, y_train)
    print('no pca forest score' + str(forest.score(X_test, y_test)))

    from sklearn.neural_network import MLPRegressor
    nn = MLPRegressor()
    nn.fit(X_train, y_train)
    print(nn.score(X_test, y_test))

    pca = PcaDecomp(df,['views', 'dislikes', 'comment_count', 'trending_time', 'category_id'])
    print(pca.get_explained_variance_ratio())

    X = pca.get_pca_df()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 69)
    forest.fit(X_train, y_train)
    print(forest.score(X_test, y_test))

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X_train, y_train)
    print(reg.score(X_test, y_test))
