import pandas as pd
import matplotlib.pylab as plt
import numpy as np

# This dataset was hosted on IBM Cloud object 
filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

# Creating of head list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

# load data
df = pd.read_csv(filename, names = headers)
print(df.head())

# Working with Missing Value
# Convert "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)

# To detect missing data
missing_data = df.isnull()
missing_data.head(5)

#Count Missing data in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    
# According to the summary above, 205 rows and 7 columns containing missing data
# as follow
# 1. "normalized-losses": 41 missing data
# 2. "num-of-doors": 2 missing data
# 3. "bore": 4 missing data
# 4. "stroke" : 4 missing data
# 5. "horsepower": 2 missing data
# 6. "peak-rpm": 2 missing data
# 7. "price": 4 missing data

# Dealing with missing data
# replace it by mean
# "normalized-losses": 41 missing data, replace them with mean
# "stroke": 4 missing data, replace them with mean
# "bore": 4 missing data, replace them with mean
# "horsepower": 2 missing data, replace them with mean
# "peak-rpm": 2 missing data, replace them with mean

# calculate the average of the "normalization-losses"column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

# Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

# Calculate the mean value for "bore" column
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

# Replace NaN by mean value
df["bore"].replace(np.nan, avg_bore, inplace=True)

# Replace NaN by mean value for "stroke" column
avg_stroke=df['stroke'].astype('float').mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

# Replace NaN by mean value for "horsepower" column
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

# Replace NaN by mean value for "peak-rpm" column
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

# Replace by frequency
# "num-of-doors": 2 missing data, replace them with "four"
# Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur
print(df['num-of-doors'].value_counts().idxmax())

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)


# Drop the whole row
# "price": 4 missing data, simply delete the whole row
# Reason: price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore any row now without price data is not useful 
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because two rows are droped
df.reset_index(drop=True, inplace=True)
print(df.head())

# Finally, the dataset is without missing values.



# Correct data format

# data types list 
print(df.dtypes)
# As the result above, some of columns are not correct data types
# "bore", "stroke", "price", "peak-rpm" should be float instead of object 
# "normalized-losses" should be int instead of object
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

print(df.dtypes)

# Data standardization
# the process of transforming data into a common format
# in order to make meaningful comparison

# Convert mpg to L/100km in "city-L/100km" column
df['city-L/100km'] = 235/df["city-mpg"]

# check the transformed data 
print(df.head())

# Convert mpg to L/100km in ""highway-mpg"" column
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"': 'highway-L/100km'}, inplace=True)

print(df.head())

# Data Normalization
# the process of transforming values of several variables into a similar range
# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
print(df[["length", "width", "height"]].head())

# Binning
# a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.
# "horsepower" is then grouped to high, medium and little horsepower

# Convert data to correct format
df["horsepower"]=df["horsepower"].astype(float, copy=True)

# plot
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()

# To separate 3 bins of equal size bandwidth
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins

# Set group name
group_names = ['Low', 'Medium', 'High']

# Apply the function "cut" the determine what each value of "df['horsepower']" belongs to
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head(20))

# See overall in each bin
print(df["horsepower-binned"].value_counts())

# plot the distribution
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()

# Bins Visulization
a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()

# Indicator variable (or dummy variable)
# the column "fuel-type" has two unique values, "gas" or "diesel"
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())

# change column's name for clarity
dummy_variable_1.rename(columns= {'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
print(dummy_variable_1.head())

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

# set the column "Aspiration-" to dummy variable
dm_var = pd.get_dummies(df['aspiration'])
dm_var.rename(columns = {'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace= True)
print(dm_var.head())

# merge the new dataframe to the original datafram
df = pd.concat([df, dm_var], axis=1)
# drop original column "aspiration" from "df"
df.drop('aspiration', axis=1, inplace=True)

# Save the new csv
df.to_csv('clean_df.csv')










