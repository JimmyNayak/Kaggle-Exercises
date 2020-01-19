import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# Get data from csv file using pandas
def init_data():
    return pd.read_csv('data_set/fifa.csv', index_col='Date', parse_dates=True)


main_data = init_data()

# print data and description about data
# print(main_data.head())
# print(main_data.describe())
# print(main_data.columns)

# Display last five data

print(main_data.tail())

# Plot the figure
plt.figure(figsize=(16, 6))
sn.lineplot(data=main_data)

plt.title("FIFA Records Analyses")
plt.show()


