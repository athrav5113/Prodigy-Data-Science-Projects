#!/usr/bin/env python
# coding: utf-8

# <center><h1>PRODIGY INFOTECH </center>
#     
# <center><h3> TASK 04 - Analyze and Visualize Sentiment patterns in Social Media</h3></center>
#     
#  ATHRAV PAWAR
#     

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


column_names = ['ID', 'entity', 'sentiment', 'comment']
df = pd.read_csv('twitter_training.csv', header=0, names=column_names)


# In[3]:


df.head()


# # EDA

# In[4]:


df.shape


# In[6]:


entity_count = df['entity'].value_counts()
print(entity_count)


# In[7]:


sentiment_count = df['sentiment'].value_counts()
print(sentiment_count)


# In[8]:


df.info


# # Checking For Duplicates

# In[9]:


duplicates = df.duplicated()
duplicated_rows = df[duplicates]
duplicated_rows.count()


# In[10]:


df.drop_duplicates(inplace=True)


# # Checking for Missing Values :

# In[11]:


df.isnull().sum()


# In[12]:


df = df.dropna()


# In[13]:


df.isnull().sum()


# In[14]:


df.nunique()


# In[15]:


for i in range(5):
    print(f"{i+1}: {df['comment'][i]}   ->   {df['sentiment'][i]}")


# # Sentiment Analysis

# In[16]:


df['sentiment'].value_counts()


# In[17]:


plt.figure(figsize=(10,5))
plt.pie(x=df['sentiment'].value_counts().values, 
        labels=df['sentiment'].value_counts().index, 
        autopct='%.1f%%', explode=[0.03, 0.03,0.03,0.03])
plt.title('The Distribution of Sentiment')
plt.show()


# In[18]:


sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()


# In[19]:


plt.figure(figsize=(15, 8))
sns.countplot(x='entity', hue='sentiment', data=df)
plt.title('Sentiment Distribution by Entity')
plt.xticks(rotation=45, fontsize=5)
plt.show()


# In[20]:


plt.figure(figsize=(15,9))
sns.barplot(x=df['entity'].value_counts().values,y=df['entity'].value_counts().index)
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('Entity Distribution')
plt.show()


# In[21]:


average_sentiment_by_entity = df.groupby('entity')['sentiment'].value_counts(normalize=True).unstack()
average_sentiment_by_entity.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Average Sentiment by Entity')
plt.xticks(rotation=45, fontsize=6)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


from wordcloud import WordCloud

positive_text = ' '.join(df[df['sentiment_score'] > 0]['comment']) 
negative_text = ' '.join(df[df['sentiment_score'] < 0]['comment'])

positive_wordcloud=WordCloud(width=800, height=400, background_color="white").generate(positive_text)
negative_wordcloud=WordCloud(width=800, height=400, background_color="black").generate(negative_text)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Positive Sentiments', fontdict={'fontsize': 20, 'color': 'black'})
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Negative Sentiments', fontdict={'fontsize': 20, 'color': 'black'})
plt.axis('off')
plt.show()


# # Positive Sentiment Word Cloud:
# 
# 

# In[32]:


from PIL import Image, ImageDraw, ImageFont


# In[33]:


positive_data = df[df['sentiment'] == 'Positive']['comment'].str.cat(sep=" ")

# Exclude the word "game" from the text data
positive_data = positive_data.replace("game", "")
if positive_data:
    wc = WordCloud(width=800, height=500, background_color='white').generate(positive_data)
    plt.figure(figsize=(12, 6))
    plt.title('Positive Sentiment Word Cloud')
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
else:
    print("No data available for positive sentiment.")


# # Negative Sentiment Word Cloud:

# In[34]:


negative_data = df[df['sentiment'] == 'Negative']['comment'].str.cat(sep=" ")

# Exclude the word "game" from the text data
negative_data = negative_data.replace("game", "")
if negative_data.strip():
    wc = WordCloud(width=800, height=500, background_color='black').generate(negative_data)
    plt.figure(figsize=(12, 6))
    plt.title('Negative Sentiment Word Cloud')
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
else:
    print("No data available for negative sentiment.")


# # Neutral Sentiment Word Cloud:

# In[35]:


neutral_data = df[df['sentiment'] == 'Neutral']['comment'].str.cat(sep=" ")

# Exclude the word "game" from the text data
neutral_data = neutral_data.replace("game", "")
if neutral_data.strip():
    wc = WordCloud(width=800, height=500, background_color='skyblue').generate(neutral_data)
    plt.figure(figsize=(12, 6))
    plt.title('Negative Sentiment Word Cloud')
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
else:
    print("No data available for neutral sentiment.")


# # Irrelevant Sentiment Word Cloud:

# In[36]:


irrelevant_data = df[df['sentiment'] == 'Irrelevant']['comment'].str.cat(sep=" ")

irrelevant_data = irrelevant_data.replace("game", "")

if irrelevant_data.strip():
    wc = WordCloud(width=800, height=400, background_color='lightpink').generate(irrelevant_data)
    plt.figure(figsize=(12, 6))
    plt.title('Irrelevant Sentiment Word Cloud')
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
else:
    print("No data available for irrelevant sentiment.")


# <h1>Conclusion</h1>
# 
# This project analyzed sentiment patterns in social media data, focusing on comments categorized by entities and sentiments. After exploring and cleaning the data, sentiment distribution was visualized, revealing proportions of positive, negative, neutral, and irrelevant sentiments. Word clouds provided insights into prevalent themes for each sentiment category. Additionally, entity-based sentiment analysis highlighted variations in sentiment across different entities mentioned in the comments. Overall, the analysis offers valuable insights into public perception, aiding businesses in understanding customer sentiment and informing strategic decisions.
