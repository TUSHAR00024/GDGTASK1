import kagglehub

# Download latest version
path = kagglehub.dataset_download("shivamb/netflix-shows")

print("Path to dataset files:", path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(path + "/netflix_titles.csv")

df.head(10)
df.shape
df.info()
df.describe(include='all')
print(f"No of duplicate `show_id` values: {df['show_id'].duplicated().sum()}")
print("there are no duplicated show_id's ")
df = df.drop('description', axis=1)
print("Dataframe after dropping 'description' column:")
df.info()
miss_count = df.isnull().sum()
print(miss_count)
df['country'].fillna('Unknown', inplace=True)
print("Missing values after")
print(df.isnull().sum())
df['director'].fillna('No Director Listed', inplace=True)
print("Missing values after")
print(df.isnull().sum())

df['duration_minutes'] = np.nan
df['seasons'] = np.nan

movie_x = df['type'] == 'Movie'
df.loc[movie_x, 'duration_minutes'] = df.loc[movie_x, 'duration'].str.extract('(\d+)', expand=False).astype(float)

tv_show_x = df['type'] == 'TV Show'
df.loc[tv_show_x, 'seasons'] = df.loc[tv_show_x, 'duration'].str.extract('(\d+)', expand=False).astype(float)

df['duration_minutes'] = df['duration_minutes'].astype('Int64')
df['seasons'] = df['seasons'].astype('Int64')

print("First 10 rows with 'duration', 'duration_minutes', and 'seasons' columns:")
display(df[['type', 'duration', 'duration_minutes', 'seasons']].head(10))

print("Info for the new duration columns:")
df[['duration_minutes', 'seasons']].info()

df['Is_Recent'] = np.where(df['release_year'] >= 2015, 1, 0)
display(df['Is_Recent'].value_counts())

display(df[['release_year', 'Is_Recent']].head(10))

plt.figure(figsize=(8, 6))
sns.countplot(x='type', data=df)
plt.title('Movies vs TV Shows')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

#Count Plot graph is for categorical data it counts the occurrence of unique data

plt.figure(figsize=(10, 6))
sns.histplot(df['release_year'], bins=30, kde=True)
plt.title('Distribution of Release Year')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.show()

#Histogram is for continuous data it groups years into "bins" to show the shape of the data.

top_10_countries = df[df['country'] != 'Unknown']['country'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_countries.index, y=top_10_countries.values)
plt.title('Top 10 Countries by Number of Releases')
plt.xlabel('Country')
plt.ylabel('Number of Releases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Bar Plot is similar to a count plot but it ranks categories by a frequency

movies_df = df[(df['type'] == 'Movie') & (df['duration_minutes'].notna())]

plt.figure(figsize=(10, 6))
sns.boxplot(x='Is_Recent', y='duration_minutes', data=movies_df)
plt.title('Movie Duration (minutes) by Recency (Recent vs. Older)')
plt.xlabel('Is Recent (0 = Older, 1 = Recent)')
plt.ylabel('Duration (minutes)')
plt.xticks(ticks=[0, 1], labels=['Older (before 2015)', 'Recent (2015 onwards)'])
plt.show()

#Boxplot Shows the Five-Number Summary (Minimum, Q1, Median, Q3, and Maximum). its is very usefull in finding outliers

numerical_features = ['release_year', 'duration_minutes', 'seasons', 'Is_Recent']
correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

#Correlation Heatmap its numbers the relation between two features the more the are related the closer to 1 and less related farther from 1 , It prevents redundancy.