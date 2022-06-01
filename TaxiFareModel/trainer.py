# imports
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

import joblib

import mlflow
from mlflow.tracking import MlflowClient

from memoized_property import memoized_property

MLFLOW_URI = 'https://mlflow.lewagon.ai/'
EXPERIMENT_NAME = '[UK][LONDON][lorcanrae] TaxiFareModel v2'

class Trainer():

    def __init__(self, X, y, experiment_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        self.mlflow_log_param('model', 'linear')

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        self.mlflow_log_metric('rmse', rmse)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



if __name__ == "__main__":
    # get data
    df_test = get_data(nrows=10_000, path='../raw_data/train.csv')
    # clean data
    df_clean = clean_data(df_test)
    # print(df_clean.head())

    # # hold out
    X_train, X_test, y_train, y_test = train_test_split(df_clean.drop('fare_amount', axis=1), df_clean['fare_amount'], test_size=0.15)

    # # set X and y
    t = Trainer(X_train, y_train, EXPERIMENT_NAME)
    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)
    # # train
    t.run()
    # # evaluate
    t.evaluate(X_test, y_test)
    # print(df_test)
    t.save_model()

