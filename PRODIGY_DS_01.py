#!/usr/bin/env python
# coding: utf-8

# <center><h1>PRODIGY INFOTECH - TASK 01</center>
#     
# <center><h3> Distribution of Categorical Variable</h3></center>
#     
#  ATHRAV PAWAR
#     

# <h3>Importing the Modules</h3>

# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# <h3>Load the data & understanding about the same</h3>

# In[2]:


data = pd.read_csv("population.csv")


# In[4]:


print(data.head(5))


# In[7]:


cols=['1960','1961','1962','1963','1964','1965','1966','1967',
      '1968','1969','1970','1971','1972','1973','1974','1975',
      '1976','1977','1978','1979','1980','1981','1982','1983',
      '1984','1985','1986','1987','1988','1989','1990','1991',
      '1992','1993','1994','1995','1996','1997','1998','1999',
      '2000','2001','2002','2003','2004','2004','2005','2006',
      '2007','2008','2009','2010','2011','2012','2013','2014',
      '2015','2016','2017','2018','2019','2020','2021','2022']


# <h3>Dealing with Null Values</h3>

# In[47]:


data.dropna()


# In[48]:


data.duplicated().sum()


# In[50]:


data.isnull().sum().any()


# In[51]:


data=data.fillna(method="ffill")
data.head()


# In[52]:


data.isnull().sum().any()


# In[73]:


data.info()


# <h3>Data Visualization</h3>

# <h3>Histogram Each Year wise - 1960 to 2022</h3>

# In[26]:


for i in cols:
    fig = plt.figure(figsize=(10, 6))  
    plt.hist(data[i], color="#2471A3", bins=5, edgecolor="black")  
    plt.xlabel(i)
    plt.ylabel("Frequency")
    plt.title("Histogram of " + i)  
    plt.tight_layout() 
    plt.show()


# <h3>Horizontal Bar - Total Count</h3>

# In[37]:


years = [str(year) for year in range(1960, 2023)]  # List of years as strings
total_values = data[years].sum()  # Sum of values for each year
colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
plt.figure(figsize=(30, 30))
plt.barh(years, total_values, color=colors)
plt.xlabel('Total Values')
plt.ylabel('Year', size=20)
plt.title('Total Count Per Year', size=20)
plt.show()


# <h1>Gender distribution Areawise</h1>

# In[61]:


data1=pd.read_csv("gender.csv")


# In[63]:


data1.shape


# In[64]:


data1.head()


# In[66]:


data1.isnull().sum()


# In[69]:


data1.describe()


# In[70]:


data1.dropna()


# In[72]:


gender_counts=data1['Region'].value_counts()
bar_width=0.9
x=range(len(gender_counts.index))

plt.bar(gender_counts.index,gender_counts.values)
plt.xlabel('Region')
plt.ylabel('Count')
plt.title("Distribution of Region")

plt.xticks(x,gender_counts.index,rotation=45)
plt.tight_layout()
plt.show()


# <h1>Conclusion</h1>
# The project provides valuable insights into population trends, gender distribution, and demographic variations over time and across regions.Further analysis could involve exploring correlations between population trends and socio-economic factors, identifying regions with rapid population growth or decline, and understanding the implications of gender distribution on various aspects of society.
# Overall, the project lays the foundation for more in-depth studies and policy-making based on population dynamics and gender demographic,Europe & Central Asia dominates while North America has the least count

# In[ ]:




