import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam

model_paths= {
    'LinearRegression': 'models/Linear_Regression_model.pkl',
    'DecisionTreeRegressor': 'models/Decision_Tree_model.pkl',
    'RandomForestRegressor': 'models/Random_Forest_model.pkl',
    'GradientBoostingRegressor': 'models/Gradient_Boosting_model.pkl',
    'XGBRegressor': 'models/XGBoost_model.pkl',
    'NeuralNetwork' : 'models/Neural_Network_model.pkl'
}

mlflow_model_paths= {
    'LinearRegression': 'runs:/1d9e376d44c94a049f97d86c8986ed5a/model',
    'DecisionTreeRegressor': 'runs:/84a78160b010499f87f4284a8cb3d52d/model',
    'RandomForestRegressor': 'runs:/bb100c3a54e54c689c18bb349cfcc9dc/model',
    'GradientBoostingRegressor': 'runs:/64aa1d95c2a143f88e859639952f3450/model',
    'XGBRegressor':'runs:/13f5b92ccd1542c3a5f26fc002571191/model'
}

class CarPricePredictionModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._prepare_data()

    def _prepare_data(self):

        self.data.fillna(method='ffill', inplace=True)
        self.data['sales_week'] = pd.to_datetime(self.data['sales_week'], format='%d%b%Y')
        self.data['year'] = self.data['sales_week'].dt.year
        self.data['month'] = self.data['sales_week'].dt.month
        self.data['day'] = self.data['sales_week'].dt.day
        self.data.drop(['sales_week'], axis=1, inplace=True)

        self.data.drop(["sales_date"], axis=1, inplace=True)
        X = self.data.drop(['price', 'transaction_id'], axis=1)
        y = self.data['price']

        categorical_columns = ['make', 'model', 'trim', 'State', 'drive_type', 'transmission', 'engine', 'bodytype']
        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
            ]
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def run_model(self, pipeline, param_grid, experiment_name):

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_

            mlflow.log_params(grid_search.best_params_)

            y_pred = best_model.predict(self.X_test)

            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = mean_squared_error(self.y_test, y_pred, squared=False)
            r2 = r2_score(self.y_test, y_pred)

            mlflow.log_metrics({
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            })

            mlflow.sklearn.log_model(best_model, 'model')
            joblib.dump(best_model, f'{experiment_name}_model.pkl')

        print(f'Best Parameters for {experiment_name}: {grid_search.best_params_}')
        print(f'MAE for {experiment_name}: {mae}')
        print(f'MSE for {experiment_name}: {mse}')
        print(f'RMSE for {experiment_name}: {rmse}')
        print(f'R2 for {experiment_name}: {r2}')


    def run_all_models(self, mode):
        
        models = {
            'Linear_Regression': (LinearRegression(), {
                'model__fit_intercept': [True, False],
            }),
            'Decision_Tree': (DecisionTreeRegressor(random_state=42), {
                'model__max_depth': [10, 20],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2]
            }),
            'Random_Forest': (RandomForestRegressor(random_state=43), {
                'model__n_estimators': [100, 200],
                'model__max_features': ['sqrt', 'log2'],
                'model__max_depth': [10, 20, 30],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            }),
            'Gradient_Boosting': (GradientBoostingRegressor(random_state=44), {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }),
            'XGBoost': (XGBRegressor(random_state=45), {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5],
                'model__subsample': [0.6, 1.0],
                'model__colsample_bytree': [0.6, 0.8]
            }),

            # 'NeuralNetwork': (KerasRegressor(model=self.build_nn_model ,  optimizer='adam', dropout_rate=0.2), {
            #     'model__batch_size': [32, 64],
            #     'model__epochs': [50, 100],
            #     'model__optimizer': ['adam', 'rmsprop'],
            #     'model__dropout_rate': [0.1, 0.2]
            # })
        }

        if mode == "all":

            for model_name, (model, param_grid) in models.items():
                pipeline = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('model', model)
                ])
                self.run_model(pipeline, param_grid, model_name)

        else : 
            
            model, param_grid = models[mode]
            
            pipeline = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('model', model)
            ])
            self.run_model(pipeline, param_grid, mode)


    def build_nn_model(self, optimizer='adam', dropout_rate=0.2):
        model = Sequential()
        model.add(Dense(64, input_dim=self.X_train.shape[1], activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def getfeatureimportance_randomforest(self, model_name):

        if model_name != 'RandomForestRegressor':
            raise ValueError('Invalid model name')
        
        logged_model = mlflow_model_paths[model_name]

        # Load model as a PyFuncModel.
        pipeline  = mlflow.sklearn.load_model(logged_model)

        model = pipeline.steps[-1][1]

        feature_importance = model.feature_importances_

        feature_importance_df = pd.DataFrame({'feature': self.X_train.columns, 'importance': feature_importance})

        return feature_importance_df.sort_values('importance', ascending=False)


    def mlflow_prediction(self, model_name, data):
        
        logged_model = mlflow_model_paths[model_name]

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Predict on a Pandas DataFrame.
        
        pred = loaded_model.predict(pd.DataFrame(data))

        pred = pd.DataFrame(pred, columns=['Price'])

        return pred


    def get_car_prediction(self, model_name, new_data):
            
            # Load the preprocessor and model
            model = joblib.load(model_paths[model_name])

            new_data.fillna(method='ffill', inplace=True)
            new_data['sales_week'] = pd.to_datetime(new_data['sales_week'], format='%d%b%Y')
            new_data['year'] = new_data['sales_week'].dt.year
            new_data['month'] = new_data['sales_week'].dt.month
            new_data['day'] = new_data['sales_week'].dt.day

            new_data.drop(['sales_week'], axis=1, inplace=True)
            new_data.drop(["sales_date"], axis=1, inplace=True)

           
            categorical_columns = ['make', 'model', 'trim', 'State', 'drive_type', 'transmission', 'engine', 'bodytype']
            numerical_columns = new_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_columns),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
                ]
            )

            preprocessed_data = preprocessor.transform(new_data)

            # Predict the price
            prediction = model.predict(preprocessed_data)
            return prediction