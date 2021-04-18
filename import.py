#import library
import pandas as pd
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(other_path, header=None)

# show the first 5 rows
print(df.head(5))

# check the bottom 10 rows 
print(df.tail(10))

# create header list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)

# replace header, then check
df.columns = headers
print(df.head(10))

# drop missing values along the column "price"
price_drop_na = df.dropna(subset=["price"], axis=0)
print(price_drop_na)


# save data set
df.to_csv("automobile.csv", index=False)
