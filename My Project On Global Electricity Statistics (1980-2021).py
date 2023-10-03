#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np 


# In[111]:


import pandas as pd 


# In[112]:


import matplotlib.pyplot as plt 


# In[113]:


data_electricity = pd.read_csv("C:\\Users\\JYOTIRMAYA SAHOO\\Downloads\\Global Electricity Statistics.csv")


# In[114]:


data_electricity.head()


# In[115]:


data_electricity.tail()


# In[116]:


data_electricity.info()


# In[117]:


data_electricity.isnull()


# In[118]:


data_electricity.isnull().sum()


# In[119]:


data_electricity['Country'] = data_electricity['Country'].str.strip()
data_electricity = data_electricity[-data_electricity['Country'].isin([
    'Micronesia', 'Northern Mariana Islands', 'Tuvalu', 'U.S. Territories', 'Reunion', 'French Guiana', 'Guadeloupe', 'Martinique'
])]

data_electricity.isnull().sum()


# In[120]:


for year in range(1980, 2022):
    data_electricity = data_electricity[-data_electricity[str(year)].isin(["--"])]


# In[121]:


data_electricity = data_electricity.melt(id_vars = ['Country', 'Features', 'Region'], var_name = 'Year', value_name = 'Value')
data_electricity.head()


# In[122]:


data_electricity['Year'] = data_electricity['Year'].astype('int')
data_electricity['Value'] = data_electricity['Value'].astype('float')


# In[123]:


data_electricity.info()


# In[124]:


data_electricity['Features'] = data_electricity['Features'].str.strip()
data_electricity = data_electricity.pivot_table(values = 'Value', index = ['Country', 'Region', 'Year'], columns = 'Features')
data_electricity.reset_index(inplace = True)


# In[125]:


data_electricity.head()


# In[126]:


import seaborn as sns


# In[127]:


import math
from scipy import stats
from scipy.stats import norm


# In[128]:


plt.figure(figsize = (5, 4), facecolor = "white")
sns.heatmap(
    data = data_electricity.corr(numeric_only = True),
    cmap = "vlag",
    vmin = -1, vmax = 1,
    linecolor = "white", linewidth = 0.5,
    annot = True,
    fmt = ".2f"
)
plt.title('Correlation Heatmap')
plt.show()


# In[129]:


def summary_numerical_dist(df_data, col, q_min, q_max):
    fig = plt.figure(figsize = (10, 8), facecolor = "white")
    layout_plot = (2, 2)
    num_subplot = 4
    axes = [None for _ in range(num_subplot)]
    list_shape_subplot = [
        [(0, 0), (0, 1), (1, 0), (1, 1)], 
        [1, 1, 1, 1],
        [1, 1, 1, 1] 
    ]
    for i in range(num_subplot):
        axes[i] = plt.subplot2grid(
            layout_plot, list_shape_subplot[0][i],
            rowspan = list_shape_subplot[1][i],
            colspan = list_shape_subplot[2][i]
        )
        sns.histplot(
        data = df_data,
        x = col,
        kde = True,
        ax = axes[0]
    )
        stats.probplot(
        x = df_data[col],
        dist = stats.norm,
        plot = axes[1]
    )
        sns.boxplot(
        data = df_data,
        x = col,
        ax = axes[2]
    )
        pts = df_data[col].quantile(q = np.arange(q_min, q_max, 0.01))
    sns.lineplot(
        x = pts.index,
        y = pts,
        ax = axes[3]
    )
    axes[3].grid(True)
    list_title = ["Histogram", "QQ plot", "Boxplot", "Outlier"]
    
    for i in range(num_subplot):
        axes[i].set_title(list_title[i])
        plt.suptitle(f"Distribution of: {col}", fontsize = 15)
    plt.tight_layout()
    plt.show()


# In[130]:


summary_numerical_dist(data_electricity, 'distribution losses', 0.95, 1)


# In[132]:


summary_numerical_dist(data_electricity, 'exports', 0.95, 1)


# In[134]:


summary_numerical_dist(data_electricity, 'imports', 0.95, 1)


# In[137]:


summary_numerical_dist(data_electricity, 'installed capacity', 0.95, 1)


# In[138]:


summary_numerical_dist(data_electricity, 'net consumption', 0.95, 1)


# In[140]:


summary_numerical_dist(data_electricity, 'net generation', 0.95, 1)


# In[141]:


summary_numerical_dist(data_electricity, 'net imports', 0.95, 1)


# In[142]:


plt.figure(figsize = (8, 6), facecolor = "white")

sns.lineplot(
    data = data_electricity,
    x = "Year", y = "losses per consumption",
    hue = 'Region',
    marker = 'o', markersize = 2
)

plt.show()


# In[ ]:




