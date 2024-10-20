# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# def preprocess_data(df):
#     df.fillna(0, inplace=True)
#     df['Year'] = df['Year'].astype(int)
#     # Add more preprocessing steps as needed
#     return df

# def train_model(df):
#     features = ['RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)', 'Year']
#     target = 'RICE PRODUCTION (1000 tons)'
    
#     X = df[features]
#     y = df[target]
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     model = RandomForestRegressor()
#     model.fit(X_train, y_train)
    
#     return model

# def evaluate_model(model, df):
#     features = ['RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)', 'Year']
#     target = 'RICE PRODUCTION (1000 tons)'
    
#     X = df[features]
#     y = df[target]
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f'Mean Squared Error: {mse}')

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def preprocess_data(df):
    df.fillna(0, inplace=True)
    df['Year'] = df['Year'].astype(int)
    # Add more preprocessing steps as needed
    return df

def train_model(df):
    features = ['RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)', 'Year']
    target = 'RICE PRODUCTION (1000 tons)'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    return model

def hyperparameter_tuning(model, X_train, y_train):
    param_grid = {
       'n_estimators': [50, 100, 200],
       'max_features': ['auto', 'sqrt', 'log2'],
       'max_depth': [None, 10, 20, 30],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def evaluate_model(model, df):
    features = ['RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)', 'Year']
    target = 'RICE PRODUCTION (1000 tons)'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Cross-validation for more robust evaluation
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f'Cross-validated Mean Squared Error: {abs(scores.mean())}')
