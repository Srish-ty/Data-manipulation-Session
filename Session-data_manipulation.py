#!/usr/bin/env python
# coding: utf-8

# # Session: Data Manipulation
# ## Introduction
# We'll be using wine-reviews dataset for today's session.
# 
# Let's have a look a our dataset
# 

# In[ ]:


import pandas as pd

wine_reviews = pd.read_csv("/content/wine-dataset.csv", index_col=0)
pd.set_option("display.max_rows", 5)


# In[ ]:


wine_reviews.head()


# In[ ]:


wine_reviews.shape


# # Checking and removing duplicates

# In[ ]:


wine_reviews.duplicated()


# In[ ]:


duplicate_rows = wine_reviews.duplicated()

duplicate_rows.head(2409)


# In[ ]:


wine_reviews[duplicate_rows]


# These are the duplicated rows.
# 43 duplicate rows.
# 
# Now let's drop these rows

# In[ ]:


wine_reviews.drop_duplicates(inplace = True)
# inplace removes in same dataset

wine_reviews


# In[ ]:


wine_reviews.shape


# # Dropping columns
# to remove unwanted features, which don't contribute to the data and might create noise.
# 
# This helps in data cleaning

# To check and review columns, let's use value_counts()

# In[ ]:


wine_reviews.taster_name


# In[ ]:


wine_reviews.taster_name.value_counts()


# In[ ]:


#wine_reviews.points

wine_reviews.price


# In[ ]:


wine_reviews.title.value_counts()

#wine_reviews.designation.value_counts()


# evidently, we don't need title column

# In[ ]:


wine_reviews.variety.value_counts()


# Let's select unwanted columns to remove:
# 
# 
# *   description
# *   region_2
# *   twitter_handle
# *   title
# 
# 
# 
# 

# In[ ]:


reviews = wine_reviews.drop(["description","region_2","taster_twitter_handle","title"], axis=1)

# or You can use inplace=True parameter

reviews


# In[ ]:


wine_reviews


# Initially there were 13 columns, out of which we've removed 4.

# ## Removing NaN values
# It's a step in data cleaning process
# 
# there are two ways to deal with NaN values
# 
# 
# 1.   removing the rows containing NaN
# 2.   replacing values with mean of that column
# 
# 
# 
# 

# In[ ]:


reviews.dropna()


# Let's fill null values of points with their min

# In[ ]:


min_points= reviews.points.min()
min_points


# In[ ]:


reviews[reviews.price.isnull()]


# In[ ]:


reviews.points.fillna(80, inplace=True)


# In[ ]:


reviews


# In[ ]:


reviews[reviews.points.isnull()]


# Let's replace null values in price column with their average

# In[ ]:


avg = reviews.price.mean()
avg


# In[ ]:


reviews.price.fillna(avg, inplace=True)


# In[ ]:


reviews[reviews.price.isnull()]


# drop rows where there's no province

# In[ ]:


reviews.province.dropna()


# In[ ]:


reviews[reviews.province.isnull()]


# In[ ]:


reviews.dropna(subset=['province', 'country'])


# ## drop the rows
# 
# let's drop rows with NaN values

# In[ ]:


reviews.dropna(inplace=True)
reviews


# As you'll learn, we do this with the `groupby()` operation.  We'll also cover some additional topics, such as more complex ways to index your DataFrames, along with how to sort your data.
# 
# # Groupwise analysis
# 
# One function we've been using heavily thus far is the `value_counts()` function. We can replicate what `value_counts()` does by doing the following:

# In[ ]:


reviews.country.value_counts()


# In[ ]:


reviews.groupby('country').price.mean()


# In[ ]:


reviews.groupby('country').price.max()


# In[ ]:


reviews[reviews.variety=='Riesling']


# In[ ]:


reviews[reviews.country=='Argentina']


# In[ ]:


reviews.groupby('winery').points.mean()


# In[ ]:


reviews.groupby('winery').points.mean().sort_values()


# In[ ]:


reviews.head()


# In[ ]:


reviews.groupby('points').points.count()


# In[ ]:


reviews.points.value_counts()


# `groupby()` created a group of reviews which allotted the same point values to the given wines. Then, for each of these groups, we grabbed the `points()` column and counted how many times it appeared.  `value_counts()` is just a shortcut to this `groupby()` operation.
# 
# We can use any of the summary functions we've used before with this data. For example, to get the cheapest wine in each point value category, we can do the following:

# In[ ]:


reviews.groupby('points').price.mean()


# You can think of each group we generate as being a slice of our DataFrame containing only data with values that match. This DataFrame is accessible to us directly using the `apply()` method, and we can then manipulate the data in any way we see fit. For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:

# In[ ]:


def show_min(arr):
  return arr.min()

show_avg = lambda arr: arr.mean()
# lambda func is a shorthand


# In[ ]:


arr1 = reviews.points

show_min(arr1)


# In[ ]:


show_avg(reviews.points)


# In[ ]:


wine_reviews.groupby('winery').apply(lambda df: df.title.iloc[0])


# In[ ]:


wine_reviews.iloc[0]


# For even more fine-grained control, you can also group by more than one column. For an example, here's how we would pick out the best wine by country _and_ province:

# In[ ]:


reviews[reviews.country=='US']


# In[ ]:


reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])


# Another `groupby()` method worth mentioning is `agg()`, which lets you run a bunch of different functions on your DataFrame simultaneously. For example, we can generate a simple statistical summary of the dataset as follows:

# In[ ]:


reviews.groupby('country').price.max()


# Aggregate method

# In[ ]:


reviews.groupby(['country']).price.agg([len, min, max])


# Effective use of `groupby()` will allow you to do lots of really powerful things with your dataset.

# # label encoding
# 
# Data can't be fed to any model with string values in it
# 
# The best way to deal with this is assigning a numerical value to the categorical values

# In[ ]:



from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()

# Fit and transform the 'Category' column to get numerical labels
reviews['country'] = label_encoder.fit_transform(reviews['country'])


# let's ahve look at new dataframe
# 

# In[ ]:


reviews


# In[ ]:


reviews.country


# In[ ]:





# # Multi-indexes
# 
# In all of the examples we've seen thus far we've been working with DataFrame or Series objects with a single-label index. `groupby()` is slightly different in the fact that, depending on the operation we run, it will sometimes result in what is called a multi-index.
# 
# A multi-index differs from a regular index in that it has multiple levels. For example:

# In[ ]:


countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed


# In[ ]:


mi = countries_reviewed.index
type(mi)


# Multi-indices have several methods for dealing with their tiered structure which are absent for single-level indices. They also require two levels of labels to retrieve a value. Dealing with multi-index output is a common "gotcha" for users new to pandas.
# 
# The use cases for a multi-index are detailed alongside instructions on using them in the [MultiIndex / Advanced Selection](https://pandas.pydata.org/pandas-docs/stable/advanced.html) section of the pandas documentation.
# 
# However, in general the multi-index method you will use most often is the one for converting back to a regular index, the `reset_index()` method:

# In[ ]:


countries_reviewed.reset_index()


# # Sorting

# In[ ]:


reviews


# In[ ]:


reviews.sort_values(by='price')


# In[ ]:


reviews.sort_values(by='points')

