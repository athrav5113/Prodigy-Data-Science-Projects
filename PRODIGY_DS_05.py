#!/usr/bin/env python
# coding: utf-8

# <center><h1>PRODIGY INFOTECH </center>
#     
# <center><h3> TASK 05 - Analyze traffic accident data to identify patterns related to road conditions, weather, and time of day. Visualize accident hotspots and contributing factors.</h3></center>
#     
#  ATHRAV PAWAR
#     

# In[1]:


# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import os
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[2]:


print(os.listdir("../input/us-accidents"))


# In[3]:


file_path = '/kaggle/input/us-accidents/US_Accidents_March23.csv'
df = pd.read_csv(file_path)
df.head()


# In[4]:


df.columns


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


#How many numerics columns here

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_df = df.select_dtypes(include=numerics)
len(numeric_df.columns)


# In[9]:


# check any missing values in columns and percentage 
missing_vales_order = df.isna().sum().sort_values(ascending = False)
missing_percentages=missing_vales_order/len(df)


# In[10]:


#here we remove missing_percentages=0 from the list
missing_percentages[missing_percentages!=0]


# In[11]:


#here we plot a Horizontal bar chart for missing_percentages>0 and think 
#about too much missing values in a colums we need to remove from our dataset

missing_percentages[missing_percentages!=0].plot(kind='barh')


# Exploratory Data Analysis
# Columns we'll analyze:
# 
# * city
# * state
# * start time
# * start lat
# * start lng
# * temprature
# * weather_condition

# In[12]:


df.columns


# In[ ]:





# # Analysis for city

# In[13]:


cities = df.City.unique()
len(cities)


# In[14]:


cities_by_accident = df.City.value_counts()
top20_city = cities_by_accident.head(20)


# In[15]:


#need to check Newyork in City
'New York' in df.City
'NY' in df.State


# In[16]:


sns.barplot(y=top20_city.keys(),x=top20_city.values)
plt.tight_layout()


# In[17]:


# last 20 cities by accident
cities_by_accident.sort_values(ascending=True).head(20)


# In[20]:


len(High_accident_cities)/len(cities)


# 8.9% of cities high acciedents 

# In[21]:


len(Low_accident_cities)/len(cities)


# # 8.9% of cities high acciedents 

# # 91% of cities Low acciedents 

# In[22]:


sns.histplot(High_accident_cities, log_scale=True,kde=True)
plt.tight_layout()


# In[23]:


sns.histplot(Low_accident_cities, log_scale=True,kde=True)


# In[24]:


cities_by_accident[cities_by_accident==1]


# In[25]:


# Over 1023 cities have reported just one accident
# so we can remove this cities from our study


# In[ ]:





# # # Analysis for Start Time

# In[26]:


# change datatime format for normal
df['Start_Time'][0]


# In[27]:


# Assuming 'df' is your DataFrame and 'Start_Time' is the column you want to convert
df.Start_Time = pd.to_datetime(df.Start_Time, errors='coerce')


# In[28]:


df.Start_Time[0].hour # change one value but we need a column

hour =df.Start_Time.dt.hour


# In[29]:


# bins 24 hrs and show percentages norm_hist
#sns.distplot(df.Start_Time.dt.hour, bins=24, kde=False,norm_hist=True)
sns.histplot(hour, color='green', bins=24, stat='percent')


# What time of the day are accidents most frequent in?
# 
# * A high percentage of accidents occur between 7 am to 8 am (probably people in a hurry to get to work)
# * Next higest percentage is 3 pm to 5 pm.

# In[30]:


day_of_week=df.Start_Time.dt.dayofweek
sns.histplot(day_of_week, color='red', bins=7, stat='percent');


# Is the distribution of accidents by hour the same on weekends as on weekdays?

# # Analysis for Start Time full Week

# In[31]:


fig, ax = plt.subplots(1, 7, figsize=(35, 6))
for i, day in enumerate(range(7)):
    day_start_time = df["Start_Time"][df["Start_Time"].dt.day_of_week == day]
    sns.histplot(day_start_time.dt.hour, color='red', bins=24, stat='percent',ax=ax[i])
    #sns.distplot(day_start_time.dt.hour, bins=24, kde=False, norm_hist=True, ax=ax[i])
    ax[i].set_title(f'Day {day}')
plt.subplots_adjust(wspace=0.2)


# In[32]:


saturdays_start_time = df.Start_Time[df.Start_Time.dt.dayofweek == 5]
sns.histplot(saturdays_start_time.dt.hour, color='red', bins=24, stat='percent')


# In[33]:


sundays_start_time = df.Start_Time[df.Start_Time.dt.dayofweek == 6]
sns.histplot(sundays_start_time.dt.hour, color='red', bins=24, stat='percent')


# * **On workings i.e. monday, tuesday, wednesday, thurday, friday you'll find almost the same trend in accidents time.
# * While on saturday and sunday the is a different trend i.e. from 10 am to 7 pm the frequency of accident is more.******

# **Analysis for Month distribtion******

# In[34]:


sns.histplot(df['Start_Time'].dt.month, color='yellow', bins=12, stat='percent')


# ****
# * The accidents are high from December and it is lowest at july. The rise continues to increase from the month of July.
# * It's seems during summer there are less accidents but as the winter starts the is a increasing trend in accidents.
# ********

# **Analysis for Year**

# In[35]:


df_2019 = df[df['Start_Time'].dt.year == 2019]
sns.histplot(df_2019['Start_Time'].dt.month, color='blue', bins=12, stat='percent')


# In[36]:


df_2020 = df[df['Start_Time'].dt.year == 2020]
sns.histplot(df_2020['Start_Time'].dt.month, color='blue', bins=12, stat='percent')


# In[37]:


df_2021 = df[df['Start_Time'].dt.year == 2021]
sns.histplot(df_2021['Start_Time'].dt.month, color='blue', bins=12, stat='percent')


# In[38]:


df_2022 = df[df['Start_Time'].dt.year == 2022]
sns.histplot(df_2022['Start_Time'].dt.month, color='blue', bins=12, stat='percent')


# In[39]:


# Much data is missing for yearly analysis 
# so,need to check some other colunm affected by our study, we can analysis source dataset


# In[40]:


df.Source


# In[41]:


df_2019 = df[df.Start_Time.dt.year == 2019]
df_2019_Source1=df_2019[df_2019.Source == 'Source1']
#sns.distplot(df_2019_Source1.Start_Time.dt.month, bins=12, kde=False,norm_hist=True)
sns.histplot(df_2019_Source1['Start_Time'].dt.month, color='blue', bins=12, stat='percent')


# In[42]:


df_2019 = df[df.Start_Time.dt.year == 2019]
df_2019_Source2=df_2019[df_2019.Source == 'Source2']
#sns.distplot(df_2019_Source2.Start_Time.dt.month, bins=12, kde=False,norm_hist=True)
sns.histplot(df_2019_Source2['Start_Time'].dt.month, color='blue', bins=12, stat='percent')


# In[43]:


df_2019 = df[df.Start_Time.dt.year == 2019]
df_2019_Source3=df_2019[df_2019.Source == 'Source3']
#sns.distplot(df_2019_Source3.Start_Time.dt.month, bins=12, kde=False,norm_hist=True)
sns.histplot(df_2019_Source3['Start_Time'].dt.month, color='blue', bins=12, stat='percent')


# In[44]:


# There seems to be some issue with the Source2 and Source3 data, need analysis these all sources now


# In[45]:


df_source = df.Source.value_counts()
df_source.plot(kind='pie')


# **There seems to be some issue with the Source2 and Source3 data so consider excluding Source2 and Source3 ******

# # **#Start Latitude & Longitude**
# 

# In[46]:


df.Start_Lat


# In[47]:


df.Start_Lng


# In[48]:


#a random sample of approximately 10% of the rows from the DataFrame df.
sample_df=df.sample(int(0.1*len(df))) 
sns.scatterplot(x=sample_df.Start_Lng, y=sample_df.Start_Lat, size=0.001)


# In[50]:


df['Temperature(F)']


# In[51]:


# Create temperature bins (customize according to your data)
bins = [0, 50, 75, 100]
labels = ['Cold', 'Moderate', 'Warm']
# Assign temperature ranges to each row
df['Temperature_Category'] = pd.cut(df['Temperature(F)'], bins=bins, labels=labels, include_lowest=True)
df['Temperature_Category']


# In[52]:


# Group by temperature category and calculate the number of accidents
#accidents_by_temperature = df.groupby('Temperature_Category', observed=False).size().reset_index(name='Accidents')

accidents_by_temperature = df['Temperature_Category'].value_counts()
df['Temperature_Category'].value_counts().plot(kind='pie')


# # Analyzing the data by state column

# In[54]:


states = df['State'].value_counts().head(5) 
states
# The data indicates california is the highest accident state 


# In[55]:


sns.barplot(y=states , x = states.index, palette="RdPu")


# <h1>Conclusion</h1>
# The analysis of accident data highlights several significant patterns and trends. Among the cities surveyed, Miami, Houston, and Los Angeles stand out with the highest reported accident rates, indicating localized areas of concern. Interestingly, the majority of cities experience relatively low accident rates, with only a minority accounting for a disproportionately high number of incidents. An intriguing observation is the presence of over 1023 cities reporting just a single accident, suggesting potential outliers that warrant further investigation. Temporally, accidents peak during morning and evening rush hours on weekdays, aligning with commuting patterns, while weekends show a different trend with increased accident frequency during daytime hours. Seasonally, accidents tend to rise as winter approaches, contrasting with lower incident rates during the summer months. Geospatial analysis reveals a concentration of accidents near bay areas, potentially influenced by population density and infrastructure. However, the absence of data from New York poses a limitation to the comprehensive understanding of nationwide accident trends. States such as California, Florida, Texas, South Carolina, and New York emerge as hotspots for accidents, highlighting the need for targeted interventions and preventive measures to ensure road safety across regions. In conclusion, this analysis underscores the importance of considering temporal, seasonal, and geographical factors in accident prevention strategies, while also emphasizing the necessity of addressing data gaps and outliers for more accurate insights and effective policymaking.
