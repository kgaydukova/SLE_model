#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # SLE_juniors 

# In[31]:


SLE_juniors = pd.read_csv('SLE_juniors.csv', sep = ';')


# In[32]:


# Remove newline characters from column names
SLE_juniors.columns = SLE_juniors.columns.str.replace('\n', '')


# In[38]:


# select columns with different dtype
int_cols = SLE_juniors.select_dtypes(include=['int64']).columns.tolist()
float_cols = SLE_juniors.select_dtypes(include=['float']).columns.tolist()
object_cols = SLE_juniors.select_dtypes(include=['object']).columns.tolist()


# In[56]:


float_cols


# In[57]:


object_cols


# In[36]:


# changing for float columns (they are int, but with a few missed values)
for x in float_cols:
    # Mode in the column
    mode_x = SLE_juniors[x].mode()[0]
    
    # Replace NaN values in column with mode of the column
    SLE_juniors.loc[SLE_juniors[x].isna(), x] = mode_x
    
    # Convert the data type of column  to int
    SLE_juniors[x] = SLE_juniors[x].astype('int')


# In[37]:


# Convert the data type of column to float
for x in object_cols:
    if (x != 'Full name'):
        #print (x, ' - ', SLE[x].isna().sum())
        # replace comma with period in column 
        SLE_juniors[x] = SLE_juniors[x].str.replace(',', '.')
        SLE_juniors[x] = SLE_juniors[x].astype(float)


# In[60]:


for x in float_cols:
    print (x, ' - ', SLE_juniors[x].isna().sum())


# In[59]:


# delete Nan rows from 'IFN-α', 'IL-18', 'IL-6'
SLE_juniors = SLE_juniors.dropna(subset=['IFN-α', 'IL-18', 'IL-6'])


# In[61]:


# delete float columns with many NaN values
float_cols = SLE_juniors.select_dtypes(include=['float']).columns.tolist()
for x in float_cols:
    if (SLE_juniors[x].isna().sum() != 0):
        print (x, ' - ', SLE_juniors[x].isna().sum())
        SLE_juniors = SLE_juniors.drop(columns=[x])


# In[62]:


SLE_juniors.dtypes.value_counts() 


# In[63]:


# select columns with different dtype
int_cols = SLE_juniors.select_dtypes(include=['int64']).columns.tolist()
float_cols = SLE_juniors.select_dtypes(include=['float']).columns.tolist()
object_cols = SLE_juniors.select_dtypes(include=['object']).columns.tolist()

# Print the float and object columns
print('int = ', len(int_cols))
print('float = ', float_cols)
print('object = ', object_cols)


# In[64]:


SLE_juniors.to_csv('SLE_juniors_clean.csv', index=False, sep = ';')


# # SLA_adults

# In[65]:


SLE_adults = pd.read_csv('SLE_adults.csv', sep = ';')


# In[66]:


# Remove newline characters from column names
SLE_adults.columns = SLE_adults.columns.str.replace('\n', '')


# In[73]:


# delete Nan rows from 'IFN-α', 'IL-18', 'IL-6'
SLE_adults = SLE_adults.dropna(subset=['IFN-α', 'IL-18', 'IL-6'])


# In[74]:


SLE_adults.dtypes.value_counts() 


# In[75]:


# select columns with different dtype
int_cols_adults = SLE_adults.select_dtypes(include=['int64']).columns.tolist()
float_cols_adults = SLE_adults.select_dtypes(include=['float']).columns.tolist()
object_cols_adults = SLE_adults.select_dtypes(include=['object']).columns.tolist()


# In[76]:


float_cols_adults


# In[77]:


object_cols_adults


# In[78]:


for x in float_cols_adults:
    print (x, ' - ', SLE_adults[x].isna().sum())


# In[79]:


for x in object_cols_adults:
    print (x, ' - ', SLE_adults[x].isna().sum())


# In[55]:


SLE_adults['ESR (in debut)'].describe()


# In[64]:


SLE_adults['ESR (in debut)'].hist()


# In[80]:


# delete float columns with many missed values 
for x in float_cols_adults:
    if SLE_adults[x].isna().sum() > 20:
        SLE_adults = SLE_adults.drop(columns=[x])


# In[81]:


float_cols_adults = SLE_adults.select_dtypes(include=['float']).columns.tolist()

# select cols for median filling missed values
cols_for_median = []
for x in float_cols_adults:
    if (SLE_adults[x].isna().sum() > 12) or (x == 'PRE (in the debut of SLE)'):
        cols_for_median.append(x)


# In[82]:


cols_for_median


# In[83]:


# select float types after drop 
float_cols_adults = SLE_adults.select_dtypes(include=['float']).columns.tolist()

# fill missed values by median or mode and change type of column to int 
for x in float_cols_adults:
    if (x in cols_for_median):
        # Median in the column
        median_x = SLE_adults[x].median()
        SLE_adults.loc[SLE_adults[x].isna(), x] = median_x
    else:
        # Mode in the column
        mode_x = SLE_adults[x].mode()[0]
        SLE_adults.loc[SLE_adults[x].isna(), x] = mode_x
    
    # Convert the data type of column  to int
    SLE_adults[x] = SLE_adults[x].astype('int')


# In[84]:


# select columns with object dtype
object_cols_adults = SLE_adults.select_dtypes(include=['object']).columns.tolist()

for x in object_cols_adults:
    print (x, ' - ', SLE_adults[x].isna().sum())


# In[75]:


SLE_adults['Anti-dsDNA (in debut)'].hist()


# In[85]:


cols_for_median_obj = []
cols_all_values = []
cols_to_delete = []

for x in object_cols_adults:
    if ((SLE_adults[x].isna().sum() == 0) and (x != 'Full name')):
        cols_all_values.append(x)       
    elif (x!= 'TBS  L1-4') and (x!= 'HDL') and (x != 'LDL') and (x != 'AC') and (x != 'Full name'): 
        cols_for_median_obj.append(x)
    elif (x != 'Full name'):
        cols_to_delete.append(x)


# In[86]:


cols_to_delete


# In[87]:


# select columns with object dtype
object_cols_adults = SLE_adults.select_dtypes(include=['object']).columns.tolist()

for x in object_cols_adults:
    
    sum_nan_for_column = SLE_adults[x].isna().sum()
    
    if (x in cols_to_delete):
        SLE_adults = SLE_adults.drop(columns=[x])
        
    elif (x in cols_for_median_obj):
        # convert column 'A' to numeric data type
        SLE_adults[x] = SLE_adults[x].str.replace(',', '.')
        SLE_adults[x] = SLE_adults[x].astype(float)

        # calculate median of column 'A'
        median_x = np.nanmedian(SLE_adults[x])

        # fill missing float values with median
        SLE_adults.loc[SLE_adults[x].isna(), x] = median_x
        
    elif (x in cols_all_values):
        # convert column 'A' to numeric data type
        SLE_adults[x] = SLE_adults[x].str.replace(',', '.')
        SLE_adults[x] = SLE_adults[x].astype(float)


# In[88]:


# select columns with different dtype
int_cols = SLE_adults.select_dtypes(include=['int64']).columns.tolist()
float_cols = SLE_adults.select_dtypes(include=['float']).columns.tolist()
object_cols = SLE_adults.select_dtypes(include=['object']).columns.tolist()

# Print the float and object columns
print('int = ', len(int_cols))
print('float = ', float_cols)
print('object = ', object_cols)


# In[89]:


SLE_adults.to_csv('SLE_adults_clean.csv', index=False, sep = ';')


# # Add classes and join two tables

# In[90]:


SLE_juniors = pd.read_csv('SLE_juniors_clean.csv', sep = ';')
SLE_adults = pd.read_csv('SLE_adults_clean.csv', sep = ';')


# In[91]:


# get the set of column names for each dataframe
cols_juniors = set(SLE_juniors.columns)
cols_adults = set(SLE_adults.columns)

# find columns that exist only in one of the dataframes
only_in_juniors = cols_juniors - cols_adults
only_in_adults = cols_adults - cols_juniors

print(f"Columns only in juniors: {only_in_juniors}")
print(f"Columns only in adults: {only_in_adults}")


# In[37]:


for x in only_in_juniors:
    SLE_juniors = SLE_juniors.drop(columns=[x])


# In[101]:


SLE_juniors.dtypes.value_counts() 


# In[102]:


SLE_adults.dtypes.value_counts() 


# In[103]:


# add a column with the same value
SLE_juniors['Class'] = 0
SLE_adults['Class'] = 1


# In[97]:


SLE_juniors['Weight'] = SLE_juniors['Weight'].astype(float)
SLE_adults['Height'] = SLE_adults['Height'].astype(float)


# In[98]:


float_cols_j = SLE_juniors.select_dtypes(include=['float']).columns.tolist()
float_cols_a = SLE_adults.select_dtypes(include=['float']).columns.tolist()


# In[99]:


float_cols_j


# In[100]:


float_cols_a


# In[104]:


# concatenate the dataframes and reset the indexes
SLE_all = pd.concat([SLE_juniors, SLE_adults], ignore_index=True)


# In[105]:


SLE_all.to_csv('SLE_all_fixed.csv', index=False, sep = ';')


# # Start with PCA

# In[2]:


SLE_param = pd.read_csv('SLE_all_fixed.csv', sep = ';')


# In[3]:


SLE_param.dtypes.value_counts() 


# In[4]:


delete_columns = ['Full name', 'DOB', 'card number']

for x in delete_columns:
    SLE_param = SLE_param.drop(columns = [x])


# In[5]:


SLE_param.to_csv('SLE_param.csv', index=False, sep = ';')


# In[145]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('sle_binary_values.csv', sep = ';')

# Separate the features and target class column
X = df.drop('Class', axis=1)
y = df['Class']

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform PCA with 2 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()

# Evaluate PCA
print('Explained variance ratio:', pca.explained_variance_ratio_)
print('Cumulative explained variance ratio:', np.cumsum(pca.explained_variance_ratio_))


# In[146]:


# calculate cumulative explained variance ratio
var_ratio = np.cumsum(pca.explained_variance_ratio_)

# plot cumulative explained variance ratio
plt.plot(range(1, len(var_ratio) + 1), var_ratio)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()


# # SLEDAY-2K 

# In[32]:


# function for sladay-2k value
def classify_sledai(value):
    if value == 0:
        return 0
    elif value >= 1 and value <= 5:
        return 1
    elif value >= 6 and value <= 10:
        return 2
    elif value >= 11 and value <= 19:
        return 3    
    else:
        return 4  


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz
import graphviz

import matplotlib.pyplot as plt # to draw graphs
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.model_selection import cross_val_score # for cross validation
from sklearn.metrics import ConfusionMatrixDisplay # creates and draws a confusion matrix


# In[91]:


# Load data
df = pd.read_csv('sle_many_values.csv', sep = ';')
df = df.drop(columns=['Class'])


# In[92]:


df['Class'] = df['SLEDAI-2K'].apply(classify_sledai)


# In[93]:


df[['SLEDAI-2K', 'Class']]


# In[94]:


df = df.drop(columns=['SLEDAI-2K'])


# In[99]:


len(df.loc[(df['Class'] == 0)])


# In[100]:


df['Class'].unique()


# In[109]:


trees = []
xtest = []
ytest = []

for i in range(100):
    # Split the dataframe into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis=1), df['Class'], test_size=0.35)

    # Fit a decision tree classifier on the training data with max_depth = 4
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)
    
    trees.append(tree)
    xtest.append(X_test)
    ytest.append(y_test)
    
    # Make predictions on the testing data
    y_pred = tree.predict(X_test)
    
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy:.2f}")
    if accuracy > 0.6:
        tree_save = tree
        name = str(i) + ' ' + str(accuracy)
        # Export the decision tree to a dot file
        dot_data = export_graphviz(tree, out_file=None, filled=True, rounded=True, feature_names=X_train.columns, class_names=['No activity', 'Low', 'Medium', 'High'], impurity=False, proportion=False, precision=0, node_ids=False)

        # Convert the dot file to PDF using Graphviz
        graph = graphviz.Source(dot_data)
        graph.render(name)


# In[110]:


## plot_confusion_matrix() will run the test data down the tree and draw
## a confusion matrix.
n_tree = 70
ConfusionMatrixDisplay.from_estimator(trees[n_tree], 
                                      xtest[n_tree], 
                                      ytest[n_tree], 
                                      display_labels=['No activity', 'Low', 'Medium', 'High'])


# In[111]:


plt.figure(figsize=(60, 30))
plot_tree(trees[70], 
          filled=True, 
          rounded=True, 
          class_names=['No activity', 'Low', 'Medium', 'High'], 
          feature_names=df.columns); 


# # k-means

# In[201]:


# Import the kmeans algorithm
from sklearn.cluster import KMeans


# In[202]:


df = pd.read_csv('sle_many_values.csv', sep = ';')
df = df.drop(columns=['Class'])


# In[203]:


df.columns


# In[204]:


first_cols = ['Age of SLE debut', 
'Disease duration',
'BMI', 
'PRE (currently)',
'Number of exacerbations of SLE',
'SF-36  (physical)',
'Er (in debut)',
'Leu  (in debut)', 
'Tr (in debut)',  
'ESR (in debut)', 
'ANF (in debut)',
'Anti-dsDNA (in debut)', 
#'IFN-α', 
'CCI', 
'CiRS (general)']


# In[205]:


first_cols


# In[206]:


for x in df.columns:
    if x not in first_cols:
        df = df.drop(columns=[x])
        print (x)


# In[207]:


df


# In[208]:


# Import the sklearn function
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled


# In[209]:


# Run a number of tests, for 1, 2, ... num_clusters
num_clusters = 5
kmeans_tests = [KMeans(n_clusters=i, init='random', n_init=10) for i in range(1, num_clusters)]
score = [kmeans_tests[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans_tests))]

# Plot the curve
plt.plot(range(1, num_clusters),score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[210]:


# Create a k-means clustering model
kmeans = KMeans(init='random', n_clusters=2, n_init=10)

# Fit the data to the model
kmeans.fit(X_scaled)

# Determine which clusters each data point belongs to:
clusters =  kmeans.predict(X_scaled)


# In[211]:


# Add cluster number to the original data
X_scaled_clustered = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
X_scaled_clustered['cluster'] = clusters

X_scaled_clustered


# In[157]:


X_scaled_clustered['cluster'].unique()


# In[158]:


len(X_scaled_clustered.loc[(X_scaled_clustered['cluster'] == 0)])


# In[159]:


len(X_scaled_clustered.loc[(X_scaled_clustered['cluster'] == 1)])


# In[136]:


len(X_scaled_clustered.loc[(X_scaled_clustered['cluster'] == 2)])


# In[160]:


X_scaled_clustered


# In[220]:


# add to original (not normalised) dataset clusters values
df_kmeans = df.assign(Class=X_scaled_clustered['cluster'].values)


# In[222]:


# make a df for 0 and 1 clusters
df_juniors = df_kmeans.loc[(df_kmeans['Class'] == 0)]
df_adults = df_kmeans.loc[(df_kmeans['Class'] == 1)]


# In[223]:


df_adults.columns


# In[231]:


df_juniors['Disease duration'].describe()


# In[232]:


df_juniors['Disease duration'].hist()


# In[234]:


df_adults['Disease duration'].describe()


# In[235]:


df_adults['Disease duration'].hist()


# In[237]:


# Calculate the median 
median = df_juniors['Age of SLE debut'].median()

# Calculate the mode 
mode = df_juniors['Age of SLE debut'].mode()[0]

print(f"Median: {median}")
print(f"Mode: {mode}")


# In[236]:


# Calculate the median 
median = df_adults['Age of SLE debut'].median()

# Calculate the mode 
mode = df_adults['Age of SLE debut'].mode()[0]

print(f"Median: {median}")
print(f"Mode: {mode}")


# In[239]:


# Calculate the median 
median = df_juniors['Disease duration'].median()

# Calculate the mode 
mode = df_juniors['Disease duration'].mode()[0]

print(f"Median: {median}")
print(f"Mode: {mode}")


# In[240]:


# Calculate the median 
median = df_adults['Disease duration'].median()

# Calculate the mode 
mode = df_adults['Disease duration'].mode()[0]

print(f"Median: {median}")
print(f"Mode: {mode}")


# In[74]:


X_scaled_clustered.to_csv('X_scaled_clustered.csv', index=False, sep = ';')


# In[228]:


df_kmeans.to_csv('kmeans_all.csv', index=False, sep = ';')
df_juniors.to_csv('kmeans_juniors.csv', index=False, sep = ';')
df_adults.to_csv('kmeans_adults.csv', index=False, sep = ';')


# In[242]:


X_scaled_clustered.columns


# In[212]:


from sklearn.decomposition import PCA

# Create a PCA model to reduce our data to 2 dimensions for visualisation
pca = PCA(n_components=2)
pca.fit(X_scaled)

# Transfor the scaled data to the new PCA space
X_reduced = pca.transform(X_scaled)


# In[213]:


# Convert to a data frame
X_reduceddf = pd.DataFrame(X_reduced, index=df.index, columns=['PC1','PC2'])
X_reduceddf['cluster'] = clusters
X_reduceddf.head()


# In[214]:


centres_reduced = pca.transform(kmeans.cluster_centers_)


# In[215]:


# Import functions created for this course
from functions import *


# In[216]:


display_factorial_planes(X_reduced, 2, pca, [(0,1)], illustrative_var = clusters, alpha = 0.8)
plt.scatter(centres_reduced[:, 0], centres_reduced[:, 1],
            marker='x', s=169, linewidths=3,
            color='k', zorder=10)


# In[217]:


df.columns


# In[247]:


# Add the cluster number to the original scaled data
X_clustered = pd.DataFrame(X_scaled, index=df.index, columns=df.columns)
X_clustered["cluster"] = clusters

# Display parallel coordinates plots, one for each cluster
display_parallel_coordinates(X_clustered, 2)


# In[248]:


df.columns


# In[249]:


# Create a data frame containing our centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)
centroids['cluster'] = centroids.index

display_parallel_coordinates_centroids(centroids, 10)


# # SLE_param dataset

# In[6]:


# Load data
df = pd.read_csv('SLE_param.csv', sep = ';')


# In[7]:


len_df = len(df)


# In[8]:


len_df


# In[9]:


# dictionary about how many unique values of features
# count_of_values : [names of features]

different_values = {}

for i in range(1, len_df + 1):
    different_values[i] = df.columns[df.nunique() == i].tolist()
    
count_of_different_values = {}
# count_of_values : [count of features]

for i in range(1, len_df + 1):
    if len(different_values[i]) != 0:
        count_of_different_values[i] = len(different_values[i])


# In[10]:


different_values


# In[11]:


# delete Class from params
different_values[2].remove('Class')
#count_of_different_values[2] = 162
count_of_different_values[2] += -1


# In[12]:


count_of_different_values[1]


# In[13]:


cols_one_value = different_values[1]
cols_binary_values = different_values[2]
cols_many_values = []

for key in different_values.keys():
    if key not in [1, 2, 3]:
        cols_many_values += different_values[key]
        
cols_one_value.append('Class')
cols_binary_values.append('Class')
cols_many_values.append('Class')


# In[15]:


cols_many_and_binary = cols_binary_values + cols_many_values


# In[16]:


cols_many_and_binary.remove('Class')


# In[17]:


cols_many_and_binary.count('Class')


# In[18]:


# select columns from dataframe using boolean mask
sle_one_value = df.loc[:, df.columns.isin(cols_one_value)]
sle_binary_values = df.loc[:, df.columns.isin(cols_binary_values)]
sle_many_values = df.loc[:, df.columns.isin(cols_many_values)]
sle_many_and_binary_values = df.loc[:, df.columns.isin(cols_many_and_binary)]


# In[19]:


# save csv with one value params, binary values and other as many values
sle_one_value.to_csv('sle_one_value.csv', index=False, sep = ';')
sle_binary_values.to_csv('sle_binary_values.csv', index=False, sep = ';')
sle_many_values.to_csv('sle_many_values.csv', index=False, sep = ';')
sle_many_and_binary_values.to_csv('sle_many_and_binary_values.csv', index=False, sep = ';')


# In[166]:


unique_values = df['Constitutional'].unique()
df[df['Constitutional'] == 4]


# In[8]:


for x in different_values[3]:
    unique_values = df[x].unique()
    print (x, ' ', unique_values)


# In[9]:


df


# # Classification Trees

# In[3]:


import matplotlib.pyplot as plt # to draw graphs
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.model_selection import cross_val_score # for cross validation
from sklearn.metrics import ConfusionMatrixDisplay # creates and draws a confusion matrix
#from sklearn.metrics import plot_confusion_matrix # to draw a confusion matrix


# In[110]:


df = pd.read_csv('sle_binary_values.csv', sep = ';')
#df = df.drop(columns=['Age of SLE debut'])
#df = df.drop(columns=['Age at diagnosis'])
#df = df.drop(columns=['Age'])
#df = df.drop(columns=['Weight'])
#df = df.drop(columns=['Height'])


# In[111]:


df.dtypes


# In[112]:


## Make a new copy of the columns used to make predictions
X = df.drop('Class', axis=1).copy() 
X.head()


# In[113]:


## Make a new copy of the column of data we want to predict
y = df['Class'].copy()
y.head()


# In[114]:


## split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

## create a decisiont tree and fit it to the training data
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)


# In[115]:


## NOTE: We can plot the tree and it is huge!
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, 
          filled=True, 
          rounded=True, 
          class_names=["Juniors", "Adults"], 
          feature_names=X.columns); 


# In[116]:


## plot_confusion_matrix() will run the test data down the tree and draw
## a confusion matrix.
ConfusionMatrixDisplay.from_estimator(clf_dt, 
                                      X_test, 
                                      y_test, 
                                      display_labels=["Juniors", "Adults"])


# In[ ]:





# In[117]:


path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # determine values for alpha
ccp_alphas = path.ccp_alphas # extract different values for alpha
ccp_alphas = ccp_alphas[:-1] # exclude the maximum value for alpha

clf_dts = [] # create an array that we will put decision trees into

## now create one decision tree per value for alpha and store it in the array
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)


# In[118]:


train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
ax.legend()
plt.show()


# In[119]:


ccp_alphas


# In[120]:


# alpha = 0.02307692


# In[121]:


clf_dt = DecisionTreeClassifier(ccp_alpha=0.023) # create the tree with ccp_alpha=0.0269

## now use 5-fold cross validation create 5 different training and testing datasets that
## are then used to train and test the tree.
## NOTE: We use 5-fold because we don't have tons of data...
scores = cross_val_score(clf_dt, X_train, y_train, cv=5) 
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})

df.plot(x='tree', y='accuracy', marker='o', linestyle='--')


# In[122]:


## create an array to store the results of each fold during cross validiation
alpha_loop_values = []

## For each candidate value for alpha, we will run 5-fold cross validation.
## Then we will store the mean and standard deviation of the scores (the accuracy) for each call
## to cross_val_score in alpha_loop_values...
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

## Now we can draw a graph of the means and standard deviations of the scores
## for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values, 
                             columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha', 
                   y='mean_accuracy', 
                   yerr='std', 
                   marker='o', 
                   linestyle='--')


# In[136]:


alpha_results[(alpha_results['alpha'] > 0.02)
              &
              (alpha_results['alpha'] < 0.04)]


# In[124]:


ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.04) 
                                & 
                                (alpha_results['alpha'] < 0.05)]['alpha']
ideal_ccp_alpha


# In[125]:


## convert ideal_ccp_alpha from a series to a float
ideal_ccp_alpha = float(ideal_ccp_alpha)
ideal_ccp_alpha


# In[139]:


## Build and train a new decision tree, only this time use the optimal value for alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=0.028)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train) 


# In[140]:


ConfusionMatrixDisplay.from_estimator(clf_dt_pruned, 
                                      X_test, 
                                      y_test, 
                                      display_labels=["Juniors", "Adults"])


# In[141]:


plt.figure(figsize=(30, 15))
plot_tree(clf_dt_pruned, 
          filled=True, 
          rounded=True, 
          class_names=["Juniors", "Adults"], 
          feature_names=X.columns); 


# In[142]:


from sklearn.tree import export_graphviz
import graphviz


# In[143]:


# Export the decision tree to a dot file
dot_data = export_graphviz(clf_dt_pruned, out_file=None, filled=True, rounded=True, feature_names=X_train.columns, class_names=['Junior', 'Adult'], impurity=False, proportion=False, precision=0, node_ids=False)

# Convert the dot file to PDF using Graphviz
graph = graphviz.Source(dot_data)
graph.render("decision_tree_pruned")


# In[ ]:




