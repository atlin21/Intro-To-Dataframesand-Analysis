import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1
rd = pd.read_csv("FloridaBikeRentals.csv", encoding='unicode_escape')
n_rows = pd.read_csv("FloridaBikeRentals.csv", encoding='unicode_escape', nrows = 3)
print(n_rows)
df1 = pd.DataFrame(n_rows)
print(df1)
# Made dataframe
df = pd.DataFrame(rd)
# finding the shape
print("Shape: ", df.shape)
# Finding column names
column_names  = df.columns
print("Column names: ", column_names)
# data types
d_type = df.dtypes
print("Datatypes: ", d_type)
# finding null values
print(df.isnull())
df.dropna()
# finding duplicates and only saving the last entry 
dupes = df.duplicated()
print(dupes)
df.duplicated(keep = 'last')
# looking at specific columns to make more efficient in memory
print(df["Temperature(°C)"])
df.astype({'Temperature(°C)': 'int32'}).dtypes
print(df["Humidity(%)"])
df.astype({'Humidity(%)': 'int32'}).dtypes
print(df["Wind speed (m/s)"])
df.astype({'Wind speed (m/s)': 'int32'}).dtypes
# convert csv to json format
df.to_json("clean.json", orient="records", indent=4)
# When converted to json format, all the rows seem to be separated into their own bundles(dictionaries). The degree symbol also is displayed as a set of "standard" characters

# Task 2
# multiply column Temperature by 10
df["Temperature(°C)"] = df["Temperature(°C)"] * 10
print(df["Temperature(°C)"])
# min max scaling
min_val = df["Visibility (10m)"].min()
max_val = df["Visibility (10m)"].max()
df["Visibility (10m)"] = (df["Visibility (10m)"] - min_val) / (max_val - min_val)
# updating json file
df.to_json("clean.json", orient="records", indent=4)
# describe function 
print(df["Temperature(°C)"].describe())
print(df["Humidity(%)"].describe())
print(df["Rented Bike Count"].describe())
print(df["Temperature(°C)"].mean())
print(df["Humidity(%)"].mean())
print(df["Rented Bike Count"].mean())
# Looking for columns who's data is valid for statistical analysis
non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
print("Non-Numeric Columns:\n", non_numeric_cols.tolist())
df.to_csv("bike_rental_processed.csv")

# Task 3
# group by season, find average of bike rented count
print(df.groupby("Seasons")["Rented Bike Count"].mean())
print(df.groupby("Holiday")["Rented Bike Count"].mean())
print(df.groupby("Functioning Day")["Rented Bike Count"].mean())
# The only trend I see is that in the winter bikes are a lot less popular

print(df.groupby(["Temperature(°C)", "Rented Bike Count"])["Hour"].mean())
print(df.groupby(["Seasons", "Rented Bike Count"])["Hour"].mean())
df.to_csv("Rental_Bike_Data_Dummy.csv")
# encoding categorical columns??
categorical_cols = df.select_dtypes(exclude=["number"]).drop(columns=["Date"]).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# Bar plot of average rented bikes per season
df["RBCM"] = df["Rented Bike Count"].mean()
season_avg_bike = df.groupby(["Seasons", "Rented Bike Count"])["Hour"].mean()
df_season_avg_bike = season_avg_bike.reset_index()
df_season_avg_bike.plot.bar(y = "Rented Bike Count", x = "Seasons")

# Task 4
# Line plot showing hourly rentals
x = df.groupby("Hour")["Rented Bike Count"].mean()
x.plot(kind="line", legend=True, marker="o", color="blue")
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Rented Bike Count", fontsize=12)
plt.plot(x)
plt.show()
# Heatmap 
correlation_matrix = df.select_dtypes(include='number').corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Matrix (df Dataset)')
plt.show()
# Temperature Boxplot
plt.figure(figsize=(12, 6))
# Box plot for Temperature and Rented Bike Count
sns.boxplot(data=df[["Temperature(°C)", "Rented Bike Count"]])
# Adding titles and labels
plt.title("Box Plot of Temperature(°C) and Rented Bike Count", fontsize=16)
plt.ylabel("Values", fontsize=12)
# Show the plot
plt.show()
# The heatmap almost creates an identity matrix and there were a lot of outliers in the "Rented Bike Count" but,
# the Temperature boxplot was pretty evenly distributed
