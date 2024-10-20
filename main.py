import pandas as pd
from utils import preprocess_data, train_model, evaluate_model

# Load the data
df = pd.read_csv('data/data.csv')

# Preprocess the data
df = preprocess_data(df)

# Train the model
model = train_model(df)

# Save the trained model
import joblib
joblib.dump(model, 'models/model.pkl')

# Evaluate the model
evaluate_model(model, df)


import matplotlib.pyplot as plt

# Plot feature importances
def plot_feature_importances(model, features):
    importance = model.feature_importances_
    feature_importances = pd.Series(importance, index=features).sort_values(ascending=False)
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances')
    plt.show()

features = ['RICE AREA (1000 ha)', 'WHEAT AREA (1000 ha)', 'Year']
plot_feature_importances(model, features)

