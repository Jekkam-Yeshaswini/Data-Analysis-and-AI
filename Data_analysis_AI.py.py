import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture as GMM


# read the appropriate data (depending on training and testing) and make DataFrame

data = pd.read_csv('training.csv')
#data = pd.read_csv('testing.csv')

# # 1. Data processing
# to print the number of unique vals
print('number of unique source IPs = ')
print(data['sourceIP'].nunique())
print('number of unique destination IPs = ')
print(data['destIP'].nunique())
print('number of unique classification = ')
print(data['classification'].nunique())

# # 2. Data analysis and visualising 

# Calc and plot the histogram for thr number of times unique source ip addresses have appeared
count_per__source_ip = data['sourceIP'].value_counts()

plt.hist(data['sourceIP'].values, bins = len(count_per__source_ip), alpha =0.5)
plt.show()

# Calc and plot the histogram for thr number of times unique source ip addresses have appeared
count_per__dest_ip = data['destIP'].value_counts()
plt.hist(data['destIP'].values, bins = len(count_per__dest_ip), alpha =0.5)
plt.show()

# # 3. Clustering

# make individual dataframe for source IP with the number of records it is apprears in.
dfs = pd.DataFrame(data=range(0,len(count_per__source_ip)), columns = ['source'])
dfs['No. Of Records (S)'] = count_per__source_ip.values


# make individual dataframe for destination IP with the number of records it is apprears in.
dfd = pd.DataFrame(data=range(0,len(count_per__dest_ip)), columns = ['dest'])
dfd['No. of Records (D)'] = count_per__dest_ip.values

#plot the individual dataframes
X = dfs.to_numpy()
Y= dfd.to_numpy()

plt.scatter(X[:,0],X[:,1], label='True Position')
plt.title('Number of records source IP apprears in')
plt.show()

#kmeans clustering for source ip
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

#print(kmeans.cluster_centers_)
#print(kmeans.labels_)
#plot with kmeans
plt.figure()
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.title('K-Means clusters for Source IP addresses')
plt.xlabel('Source IP adresses representations')
plt.ylabel('Number of record appearences')
plt.show()

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean')**2, axis=1)) / X.shape[0])
#print(X)
# Plot the elbow
plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k for source ip clusters')
plt.show()


#kmeans clustering for dest ip

plt.scatter(Y[:,0],Y[:,1], label='True Position')
plt.title('Number of records dest IP apprears in')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(Y)

#print(kmeans.cluster_centers_)
#print(kmeans.labels_)


#plot with kmeans
plt.figure()
plt.scatter(Y[:,0],Y[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.title('K-Means clusters for Destinations IP addresses')
plt.xlabel('Destinations IP adresses representations')
plt.ylabel('Number of record appearences')
plt.show()

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(Y)
    kmeanModel.fit(Y)
    distortions.append(sum(np.min(cdist(Y, kmeanModel.cluster_centers_,'euclidean')**2, axis=1)) / Y.shape[0])



# Plot the elbow
plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k for dest ip clusters')
plt.show()

# Hierachical clustering for source IP


linked = linkage(X, 'single')
labelList = range(0, len(X))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title('Source IP clustered Hierachically by min dist')
plt.show()

linked = linkage(X, 'complete')
labelList = range(0, len(X))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title('Source IP clustered Hierachically by max dist')
plt.show()

linked = linkage(X, 'average')
labelList = range(0, len(X))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title('Source IP clustered Hierachically by average dist')
plt.show()

linked = linkage(X, 'centroid')
labelList = range(0, len(X))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title('Source IP clustered Hierachically by centeroid dist')
plt.show()


# Hierachical clustering for dest IP

linked = linkage(Y, 'single')
labelList = range(0, len(Y))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title('dest IP clustered Hierachically by min dist')
plt.show()

linked = linkage(Y, 'complete')
labelList = range(0, len(Y))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title('Dest IP clustered Hierachically by max dist')
plt.show()

linked = linkage(Y, 'average')
labelList = range(0, len(Y))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title('Dest IP clustered Hierachically by average dist')
plt.show()

linked = linkage(Y, 'centroid')
labelList = range(0, len(Y))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title('Dest IP clustered Hierachically by centeroid dist')
plt.show()

#EM algorithms for source ip
gmm = GMM(n_components=3).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.title(' EM algorithm for source IP addresses')
plt.show()

#EM algorithms for dest ip
gmm = GMM(n_components=2).fit(Y)
labels = gmm.predict(Y)
plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=40, cmap='viridis')
plt.title(' EM algorithm for Destination IP addresses')
plt.show()

# # 4. Finding relationships

# data of both source and destination ip adresses with number of records they appear in.
dfs = pd.DataFrame(data=(count_per__source_ip.index), columns=['Source IP'])
dfs['No. of Records (S)'] = count_per__source_ip.values


dfd = pd.DataFrame(data=(count_per__dest_ip.index), columns=['destination IP'])
dfd['No. of Records (D)'] = count_per__dest_ip.values


# section the source and destination ip adresses depending on how many records they appear in.
S = dfs['Source IP'].tolist()
S_A = dfs.loc[dfs['No. of Records (S)']<21, 'Source IP'].tolist()
S_B = dfs.loc[np.logical_and(dfs['No. of Records (S)']>20, dfs['No. of Records (S)']<201), 'Source IP'].tolist()
S_C = dfs.loc[np.logical_and(dfs['No. of Records (S)']>200, dfs['No. of Records (S)']<401), 'Source IP'].tolist()
S_D = dfs.loc[dfs['No. of Records (S)']>400, 'Source IP'].tolist()


D = dfd['destination IP'].tolist()
D_A = dfd.loc[dfd['No. of Records (D)']<41, 'destination IP'].tolist()
D_B = dfd.loc[np.logical_and(dfd['No. of Records (D)']>40, dfd['No. of Records (D)']<101), 'destination IP'].tolist()
D_C = dfd.loc[np.logical_and(dfd['No. of Records (D)']>100, dfd['No. of Records (D)']<401), 'destination IP'].tolist()
D_D = dfd.loc[dfd['No. of Records (D)']>400, 'destination IP'].tolist()



# Delete unwanted columns of the data frame.

labels = ["priority", "label", "packet info", "packet info cont'd", "xref","sourcePort", "destPort"]
data = data.drop(columns = labels)

#assign cluster number for all source ip addresses

source_cluster = []
sdata= data['sourceIP']
for i in range(0,len(data['sourceIP'])):
    if sdata[i] in S_A:
        X=1
    if sdata[i] in S_B:
        X=2
    if sdata[i] in S_C:
        X=3
    if sdata[i] in S_D:
        X=4
    source_cluster.append(X)

data["source cluster number"] = source_cluster

# assign cluster numbers for all destination ip adresses.
dest_cluster = []
ddata= data['destIP']
for i in range(0,len(data['destIP'])):
    if ddata[i] in D_A:
        X=1
    if ddata[i] in D_B:
        X=2
    if ddata[i] in D_C:
        X=3
    if ddata[i] in D_D:
        X=4
    dest_cluster.append(X)

data["destination cluster number"] = dest_cluster


#relationship between source bin 1 to destination bins.
rel_var1 = data[data['source cluster number'] == 1]
source_var1 = rel_var1['destination cluster number'].value_counts()

row = pd.Series([0,0,0,0,0])
source_var1= pd.concat([source_var1, row], axis=1)
source_var1 = source_var1.drop(columns=[0])


#relation ship between source bin 2 to destination bins.
rel_var2 = data[data['source cluster number'] == 2]
source_var2 = rel_var2['destination cluster number'].value_counts()
source_var2= pd.concat([source_var2, row], axis=1)
source_var2 = source_var2.drop(columns=[0])
#print(source_var2)

#relation ship between source bin 3 to destination bins.
rel_var3 = data[data['source cluster number'] == 3]
source_var3 = rel_var3['destination cluster number'].value_counts()
source_var3= pd.concat([source_var3, row], axis=1)
source_var3 = source_var3.drop(columns=[0])
#print(source_var3)

#relation ship between source bin 4 to destination bins.
rel_var4 = data[data['source cluster number'] == 4]
source_var4 = rel_var4['destination cluster number'].value_counts()
source_var4= pd.concat([source_var4, row], axis=1)
source_var4 = source_var4.drop(columns=[0])
#print(source_var4)

#create another dataframe to calculate the conditional probabilities


Conditional_probabilities = { 'D1':[source_var1['destination cluster number'].loc[1],source_var2['destination cluster number'].loc[1],source_var3['destination cluster number'].loc[1],source_var4['destination cluster number'].loc[1]],
                              'D2':[source_var1['destination cluster number'].loc[2],source_var2['destination cluster number'].loc[2],source_var3['destination cluster number'].loc[2],source_var4['destination cluster number'].loc[2]],
                              'D3':[source_var1['destination cluster number'].loc[3],source_var2['destination cluster number'].loc[3],source_var3['destination cluster number'].loc[3],source_var4['destination cluster number'].loc[3]],
                              'D4':[source_var1['destination cluster number'].loc[4],source_var2['destination cluster number'].loc[4],source_var3['destination cluster number'].loc[4],source_var4['destination cluster number'].loc[4]],
                              's':['S1','S2','S3','S4']}
Conditional_probs = pd.DataFrame(Conditional_probabilities)

Conditional_probs = Conditional_probs.set_index('s')
total = Conditional_probs.sum(axis = 1)

Conditional_probs = Conditional_probs.T

#calculate conditional probabilities

Conditional_probs['S1'].loc['D1'] = (Conditional_probs['S1'].loc['D1']/total.loc['S1'])*100
Conditional_probs['S1'].loc['D2'] = (Conditional_probs['S1'].loc['D2']/total.loc['S1'])*100
Conditional_probs['S1'].loc['D3'] = (Conditional_probs['S1'].loc['D3']/total.loc['S1'])*100
Conditional_probs['S1'].loc['D4'] = (Conditional_probs['S1'].loc['D4']/total.loc['S1'])*100

Conditional_probs['S2'].loc['D1'] = (Conditional_probs['S2'].loc['D1']/total.loc['S2'])*100
Conditional_probs['S2'].loc['D2'] = (Conditional_probs['S2'].loc['D2']/total.loc['S2'])*100
Conditional_probs['S2'].loc['D3'] = (Conditional_probs['S2'].loc['D3']/total.loc['S2'])*100
Conditional_probs['S2'].loc['D4'] = (Conditional_probs['S2'].loc['D4']/total.loc['S2'])*100

Conditional_probs['S3'].loc['D1'] = (Conditional_probs['S3'].loc['D1']/total.loc['S3'])*100
Conditional_probs['S3'].loc['D2'] = (Conditional_probs['S3'].loc['D2']/total.loc['S3'])*100
Conditional_probs['S3'].loc['D3'] = (Conditional_probs['S3'].loc['D3']/total.loc['S3'])*100
Conditional_probs['S3'].loc['D4'] = (Conditional_probs['S3'].loc['D4']/total.loc['S3'])*100

Conditional_probs['S4'].loc['D1'] = (Conditional_probs['S4'].loc['D1']/total.loc['S4'])*100
Conditional_probs['S4'].loc['D2'] = (Conditional_probs['S4'].loc['D2']/total.loc['S4'])*100
Conditional_probs['S4'].loc['D3'] = (Conditional_probs['S4'].loc['D3']/total.loc['S4'])*100
Conditional_probs['S4'].loc['D4'] = (Conditional_probs['S4'].loc['D4']/total.loc['S4'])*100

print(Conditional_probs)

# plot the conditional probabilities
fig, axes = plt.subplots(nrows=1,ncols=4)
fig.suptitle('Conditional probabilities of destination clusters being contacted given source clusters respectively')

Conditional_probs.plot(y = ['S1'], ax = axes[0], kind='bar', subplots=True, legend = False)
Conditional_probs.plot(y = ['S2'], ax = axes[1], kind='bar', subplots=True, legend = False)
Conditional_probs.plot(y = ['S3'], ax = axes[2], kind='bar', subplots=True, legend = False)
Conditional_probs.plot(y = ['S4'], ax = axes[3], kind='bar', subplots=True, legend = False)
plt.show()

# #  5: Decision trees

global layer
layer = 0

# Necessary functions for implementations


#it returns the entropy of the given data.
def calculate_entropy(data):
    label_column=data['classification']
    _,counts = np.unique(label_column, return_counts=True)

#    print(counts)
    probabilities = counts/counts.sum()

    entropy = sum(probabilities *  -np.log2(probabilities))
#    entropy = 1-sum(probabilities*probabilities)
    return entropy

#given the data and the attribute along with the cluster number, it returns a subframe
def sub_dataframes(data, attribute, cluster):

    subframe = data[data[attribute]==cluster]

    return subframe

# finds the best attribute by comparing the information gain of each and returns the attribute
def det_best_attribute(data, attribute1, attribute2):
    cluster_att_1 = []
    cluster_att_2 = []
    cluster_att_1 =  np.unique(data[attribute1].unique())
    cluster_att_2 =  np.unique(data[attribute2].unique())

    E_h1 = 0.0000;
    E_h2 = 0.0000;
    for i in cluster_att_1:
        E_h1 = E_h1 + (calculate_entropy(sub_dataframes(data,attribute1,i))*(len(sub_dataframes(data, attribute1, i))/len(data)))

    IG_1 = calculate_entropy(data) - E_h1

    for i in cluster_att_2:
        E_h2 = E_h2 + (calculate_entropy(sub_dataframes(data,attribute2,i))*(len(sub_dataframes(data, attribute2, i))/len(data)))

    IG_2 = calculate_entropy(data) - E_h2

    if IG_1 == max(IG_1, IG_2):
        #print(" Information gain1: ", IG_1)
        #print(" Information gain2: ", IG_2)
        return attribute1
    elif IG_2 == max(IG_1, IG_2):
        #print(" Information gain1: ", IG_1)
        #print(" Information gain2: ", IG_2)
        return attribute2
    return

'''
splits the data given based on the attribute passed and also subdivides this into the clusters'.
It then evaluates whether or not it leads to a pure or impure node. depending on which, it predicts
the class. (for pure node, the prediction class is obvious. and for the terminated impure node, the prediction
class is set at the majority of current classes)
'''
def split_data(data_now, best_attribute):
    attribute_list.remove(best_attribute)
    print("Node Attribute: ", best_attribute, "  (", len(data_now), ") ")
    print("  ")
    cluster = np.unique(data_now[best_attribute].unique())
    global expand_vars
    expand_vars = []
    global layer
    layer = layer +1
    for i in cluster:

        node = sub_dataframes(data_now,best_attribute,i)
        classify_index = data_now[data_now[best_attribute] == i].index.values

        if node['classification'].nunique() == 1:
            classify_with = node['classification'].value_counts()
            data.loc[classify_index,['predicted class']] = classify_with.index[0]
            print('Node cluster',i, '  (', len(node), ')  pure classification: ', classify_with.index[0])
        else:
            expand_vars.append(i)
            print('Node cluster',i, '  (', len(node), ')  impure classification: ', 'EXPAND')
            if len(attribute_list) == 0 :
                classify_with = data['classification'].value_counts().idxmax()
                data.loc[classify_index,['predicted class']] = classify_with
                print("                 End node, cannot expand.", "Majority class is: ", classify_with)
                if (layer ==2) & (len(data) >= 20000):
                    attribute_list.append(best_attribute)

    print("   ")
    return data

'''
The impure nodes get to expand the tree further based on the next attribute.
It is done in this function given the data and variable points that need to be expanded.
'''
# expand the tree
def expand(data, expand_variable):
    for i in expand_variable:
        next_node = sub_dataframes(data, root_attribute, i)
        split_data(next_node, det_best_attribute(next_node, 'source cluster number', 'destination cluster number'))
        #print(' ',det_best_attribute(next_node, 'source cluster number', 'destination cluster number') )
    return

# it calculates the percentage of values that the prediction is correct for.
def calc_accuracy(data):
    acc = 0
    for i in range(0,len(data)):

        if data.loc[i,['classification']].values == data.loc[i,['predicted class']].values:
            acc = acc + 1
    acc = (acc/total_data_points)*100
    print("The decision tree gives an unambiguous answer", acc, " percent of the time")
    return acc



# sequence the decision tree to display it.
attribute_list = ['source cluster number', 'destination cluster number']
root_attribute = det_best_attribute(data,'source cluster number', 'destination cluster number')
total_data_points = len(data)

empty_array = np.empty(len(data))
empty_array[:] = np.nan
data['predicted class'] = empty_array

print("   ")
print("   ")
print("       DECISION TREE       ")
print("   ")
print("   ")

split_data(data,det_best_attribute(data,'source cluster number', 'destination cluster number'))
current_expand_var = expand_vars
expand(data, current_expand_var)

print("   ")
print("   ")

calc_accuracy(data)
