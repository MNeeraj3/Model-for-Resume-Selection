### Problem Statement:

#### Build a Machine Learning model , which will help the management to Identify the potential candidates (good candidate to offer job) for both the roles (Web Development and Data Scientist) from the given data.

### Solution Approch:

# * Since it is unlabbeled data, I am approching by unsuperwise Algorithms
# * For Other Skill to take into the model, doing text analysis
# * Splitting the Other Skill into core skills of Data Science and Web Developement
# * Providing the Weight to Given skills, Other Skills and Academics
# * Using K-mean clustering to make clusters

#importing the required Library
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# reading data set
data=pd.read_excel('mavoix_ml_sample_dataset.xlsx')
df = data.copy() # making a copy 
df.head(3) # top 3 rows

# data information
df.info()

df.shape

## Null Values


# Null value counts
df.isna().sum()

# * Unnamed: 10 completely null Deleting this column
# * Instead of using the stream feature, considering Skills related to Data Science and Web Devep.
# * In Performance in PG/UG/12th/10th giving 1 if Score is First Devision 0 otherwise
# * In Degree if they are PG then providing 1 else 0


# Deleting the column "Unnamed: 10"
del df['Unnamed: 10']

# Filling all the missing values with zero to perform the above actions
df.fillna(0,inplace = True)

#### Handling Others Skills

# converting type into list of string
df['Other skills']  = df['Other skills'].apply(lambda x: str(x).split(", "))

# All skill Set
skill_set = set(df['Other skills'].sum())
print(skill_set)

# Other Core Skills for Web Developement
other_core_web = {'HTML', 'SAP', 'User Interface (UI) Development', 'Search Engine Optimization (SEO)', 'Apache', 'Linux', 'Hadoop', 'Bootstrap'}

# Other Core Skills for Data Science
other_core_DataScience = {'SAS', 'Data Science', 'Natural Language Processing (NLP)', 'BIG DATA ANALYTICS', 'Statistical Modeling','Neural Network', 'Machine Learning'}

# creating two new features
df['other_DS_skills(out_6)'] = df['Other skills'].apply(lambda x: len(set(x).intersection(other_core_DataScience)))
df['other_WD_skills(out_8)'] = df['Other skills'].apply(lambda x: len(set(x).intersection(other_core_web)))

# handelling UG/PG Degree
df['Degree'] = df.Degree.apply(lambda x: str(x).split(" ")[0])
df['Degree'] = df['Degree'].replace({'B.Com.':0, 'B.Tech':0,'Bachelor':0, 'Executive':0, 'Integrated':1,
                      'MBA':1, 'Master':1, 'PG':1, 'Post':1, '0':0})

# functions for transforming PG/UG/12th/10th features
def division_graduation(x):
    if x != 0:
        temp = str(x).split('/')
        x = float(temp[0])
        total= int(temp[-1])
        if x/total > 0.6:
            return 1
    return 0
def performance(df,col):
    for i in col:
        df[i] = df[i].map(division_graduation)
    return df

def division_intermediate(x):
    if x > 10:
        x /= 10
    if x>6:
        return 1
    return 0
    

df['Performance_12'] =  df['Performance_12'].apply(lambda x:division_intermediate(float(str(x).split("/")[0])))
df['Performance_10'] =  df['Performance_10'].apply(lambda x:division_intermediate(float(str(x).split("/")[0])))

df = performance(df, ['Performance_PG', 'Performance_UG'])
df['year_gap'] = df['Current Year Of Graduation'] - 2020

# Dropping all the object columns
df = df.select_dtypes(exclude=object)

## Scores

# > Providing manual Weights, 
# * 70% to Given Skills
# * 20% to Other Skills
# * 10% to Academics

df['DS_SCORE'] = 0.7*(df[['Python (out of 3)','R Programming (out of 3)', 'Deep Learning (out of 3)']].\
                      sum(axis = 1)) + 0.20*(df['other_DS_skills(out_6)']) +\
                    0.10*(df[['Degree','Performance_PG', 'Performance_UG', 'Performance_12', 'Performance_10']].\
                          sum(axis = 1))

df['WD_SCORE'] = 0.7*(df[['PHP (out of 3)', 'MySQL (out of 3)', 'HTML (out of 3)', 'CSS (out of 3)',
                          'JavaScript (out of 3)', 'AJAX (out of 3)', 'Bootstrap (out of 3)', 
                          'MongoDB (out of 3)', 'Node.js (out of 3)','ReactJS (out of 3)']].\
                      sum(axis = 1)) + 0.20*(df['other_WD_skills(out_8)']) +\
                0.10*(df[['Degree','Performance_PG', 'Performance_UG', 'Performance_12',
                          'Performance_10']].sum(axis = 1))

## Clustering Using K-Mean Clustering

# create kmeans object for data science
kmeans_ds = KMeans(n_clusters=2, random_state=42)
# fit kmeans object on  data data science score
kmeans_ds.fit(np.array(df['DS_SCORE']).reshape(-1,1))
# df['DS_SCORE_cluster'] = kmeans_ds.labels_
cen_ds = kmeans_ds.cluster_centers_

# create kmeans object for Web Developer 
kmeans_wd = KMeans(n_clusters=2, random_state=42)
# fit kmeans clustering on Web Developer Score
kmeans_wd.fit(np.array(df['WD_SCORE']).reshape(-1,1))
cen_wd = kmeans_wd.cluster_centers_
# df['WD_SCORE_cluster'] = kmeans_wd.labels_

print("WD Centroid:", cen_wd)
print("DS Centroid:", cen_ds)

def lable(centroid, points):
    if abs(max(centroid)-points) < abs(min(centroid)-points):
        return 1
    return 0

df['DS_SCORE_cluster'] = df['DS_SCORE'].apply(lambda x: lable(cen_ds,x))
df['WD_SCORE_cluster'] = df['WD_SCORE'].apply(lambda x: lable(cen_wd,x))

data['Suitablefor'] = df[['DS_SCORE_cluster','WD_SCORE_cluster']].apply(lambda x: 1 if (x['DS_SCORE_cluster']==1) & (x['WD_SCORE_cluster']==1) else 0, axis=1)


def suitable(x):
    if (x.loc['DS_SCORE_cluster']==1) & (x.loc['WD_SCORE_cluster']==0):
        return 'Data Scientist'
    if (x.loc['DS_SCORE_cluster']==0) & (x.loc['WD_SCORE_cluster']==1):
        return 'Web Developer'
    if (x.loc['DS_SCORE_cluster']==1) & (x.loc['WD_SCORE_cluster']==1):
        return 'Both(Data Science & Web Developer)'
    return 'Not Eligible'
data['Suitable_for'] = df[['DS_SCORE_cluster','WD_SCORE_cluster']].apply(lambda x: suitable(x),axis =1)


#### Some Output insight

Candidate_for_DS_n_WD = data[data['Suitable_for'] == 'Both(Data Science & Web Developer)'].shape[0]
print("Candidate for Data Science & Web Dev both: {} out of {}".format(Candidate_for_DS_n_WD,data.shape[0]))

Candidate_for_DS = data[data['Suitable_for'] == 'Data Scientist'].shape[0]
print("Candidate for Data Science: {} out of {}".format(Candidate_for_DS,data.shape[0]))

Candidate_for_WD = data[data['Suitable_for'] == 'Web Developer'].shape[0]
print("Candidate for Web Developer: {} out of {}".format(Candidate_for_WD,data.shape[0]))

Candidate_NE = data[data['Suitable_for'] == 'Not Eligible'].shape[0]
print("Candidates that are Not Eligible: {} out of {}".format(Candidate_NE,data.shape[0]))

# Saving as a dataframe
data.to_csv("Output.csv")
