import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read train CSV file from the data set
home_data = pd.read_csv("data_set/train.csv")
# print(home_data.head())
# print(home_data.columns)

# Define the target(in this case house price is the target)
target_price = home_data.SalePrice
# print(home_data.SalePrice)

# Define the features(in this case columns which can cause or predict the price)
features = ['Id', 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr',
            'TotRmsAbvGrd']

features_x = home_data[features]

# print(features_x)

# Split data into train and test for validation
train_x, test_x, train_y, test_y = train_test_split(features_x, target_price, random_state=1)

# Specify the model and fit the data
# model = DecisionTreeRegressor(random_state=1)
# model.fit(train_x, train_y)
#
# # Make validation predictions and calculate mean absolute error
# model_prediction = model.predict(test_x)P
# # print(model_prediction)
#
# val_mae = mean_absolute_error(model_prediction, test_y)
# print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
#
# # Using best value for max_leaf_nodes
# model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
# model.fit(train_x, train_y)
#
# # Make validation predictions and calculate mean absolute error
# model_prediction = model.predict(test_x)P
# # print(model_prediction)
#
# val_mae = mean_absolute_error(model_prediction, test_y)
# print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Random forest model
model = RandomForestRegressor(random_state=1)
model.fit(train_x, train_y)


# Make validation predictions and calculate mean absolute error
model_prediction = model.predict(test_x)
# print(model_prediction)
val_mae = mean_absolute_error(model_prediction, test_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(val_mae))

# print(len(model_prediction))
# print(len(test_x))

print(test_x.columns)

output = pd.DataFrame({'Id': test_x.Id, 'SalePrice': model_prediction})
output.to_csv('submission.csv', index=False)
