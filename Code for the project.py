@author: Ellie
"""

import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split
import gower
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from sklearn.manifold import TSNE

###view treatment
drug = pd.read_csv('directory/id_tx.csv') #contains id and treatment for each patient
dclass = pd.read_csv('directory/tx_class.csv') #lists of drug and corresponding drug classes
dclass = dclass[['drug','drug class']]

id_drug = pd.merge(drug, dclass, on = 'drug')
id_drug = id_drug.dropna(subset = ['drug class'])
id_drug = id_drug.drop(['drug'],axis = 1)
id_drug = id_drug.drop_duplicates() #remove duplicate

#pivot
id_drug['values'] = 1
pivot_class = id_drug.pivot(index = 'hadm_id', columns = 'drug class', values = 'values')
pivot_class = pivot_class.fillna(0) #if not prescribed with drug, = 0

countp = pivot_class.drop_duplicates()# =783


data = pd.read_csv('directory/features 1477.csv',index_col='HADM_ID') #contains id and features for each patient
data.head() 

##data pre-processing
#
#
#
#

#missing values
data.isnull().sum()
null_data = data[data.isnull().any(axis = 1)]

#NTproBNP and lactate removed as it has missing values > 25%; BG = glucose
data.drop(['Lactate','NTproBNP','BG'],inplace = True, axis = 1)

#outlier detection according YerevaNN/mimic3-benchmarks
data.min()
data.max()
#no outlier detected

#replace missing values
newd = data.copy()
#newd.drop('HADM_ID', inplace = True, axis =1)
newd.loc[newd['admission_age'] > 299, 'admission_age'] = 90

#only keep the index with treatment
newd = newd[newd.index.isin(pivot_class.index)]

##select continuous variable
colnum = list(range(3,34))
colnum.append(0)
colnum.sort()
convar=newd[newd.columns[colnum]] 

##############################################################
###replace missing value
xdata = convar.copy()
#change LVEF to 0-4, 4 being severe
xdata['LVEF'] = xdata['LVEF'].replace(['Hyperdynamic','Normal','Mild','Moderate','Severe'], [0,1,2,3,4])
ndistvar = [2,3,4,6,8,9,10,12,13,14,15,16,24,25,26,27,28,29]
xdata.fillna(xdata.iloc[:,ndistvar].mean(),inplace = True) #continuous variable with mean
xdata.fillna(xdata.iloc[:,~ xdata.columns.isin(xdata.columns[ndistvar])].median(),inplace = True) #assym with median
contvar = xdata.copy()

#categorical variables
catvar = newd[['ethnicity','gender','hypertension','blood_loss_anemia','cardiac_arrhythmias', 'chronic_pulmonary', 'deficiency_anemias' ,'obesity','renal_failure','psychoses','depression','diabetes_complicated','diabetes_uncomplicated']]
catvar['gender'] = catvar['gender'].replace(['M','F'],[1,0]) #change M =1, F=0
catvar['diabetes']=catvar['diabetes_uncomplicated']
catvar.loc[catvar['diabetes_complicated'] ==1, 'diabetes'] = 1
catvar.drop(['diabetes_uncomplicated','diabetes_complicated'],inplace = True, axis = 1)

dummies = pd.get_dummies(catvar['ethnicity'],prefix="eth")
catvar      = pd.concat([catvar,dummies], axis=1)
del catvar['ethnicity']

xdata = pd.concat([xdata,catvar],axis=1)

#split data

train,test = train_test_split(xdata, test_size = 0.3, random_state = 1)#1，1000, 1999，2022,2000, 1077, 1000

train_tx = pivot_class[pivot_class.index.isin(train.index)]
test_tx = pivot_class[pivot_class.index.isin(test.index)]
tx_perc = round(train_tx.mean()*100,2)

#characteristic comparison
#continuous variables normal - unpaired t-test
from scipy import stats
for nvar in ndistvar:
    i = train.columns[nvar]
    print(i)
    print(stats.ttest_ind(train[i], test[i]))
#continuous variables not normal - unpaired t-test
notnvar = contvar.columns[~contvar.columns.isin(contvar.columns[ndistvar])]

xdata['admission_age'].median()
for i in notnvar:
    print(i)
    statss, p = stats.ranksums(train[i],test[i])
    print(p)
#categorical 
#draw contigency table
catname = list(catvar.columns)

for i in notnvar:
    print(i,xdata[i].quantile(0.25),xdata[i].quantile(0.75))
xdata[xdata['gender']==0].count()

for i in catname:
    contingency_table = [[],[]]
    for j in np.unique(catvar[i]):
        contingency_table[0].append(train[train[i] == j].shape[0])
        contingency_table[1].append(test[test[i] == j].shape[0])
        print(j)
    print(contingency_table)
    stat,p,dof,expected = stats.chi2_contingency(contingency_table)
    print(i,p)

#feature selection + scaling (everything scaled to 0-1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
scaler = MinMaxScaler()
tr_n = scaler.fit_transform(train)
tr_n = pd.DataFrame(tr_n, columns = train.columns, index = train.index)

### feature seletion with k-medoids
maxvars = 20
kmin = 3
kmax = 14
cols = list(tr_n.columns)
#
kmed_kwargs = {"init": "random","max_iter": 1000,"random_state": 1996, "metric":'manhattan'}
cut_off = 0.5
results_for_each_k = []
vars_for_each_k = {}

####k-medoids for mixed data type
##feature selection using k-medoid
for k in range(kmin,kmax+1):
    selected_variables=cols.copy()
    while(len(selected_variables)>maxvars):
        results=[]
        for col in selected_variables:
            scols=selected_variables.copy()
            scols.remove(col) 
            kmedoids = KMedoids(n_clusters=k, **kmed_kwargs).fit(tr_n[scols])
            results.append(silhouette_score(tr_n[scols], kmedoids.predict(tr_n[scols]),metric = 'manhattan'))
        selected_var=selected_variables[np.argmin(results)]
        selected_variables.remove(selected_var)
    results_for_each_k.append(max(results))
    vars_for_each_k[k]=selected_variables

##select the constant pool
vars_count = {}
vars_list = sum(vars_for_each_k.values(), [])
for i in np.unique(vars_list):
    vars_count[i] = vars_list.count(i)
sorted(vars_count.items(), key=lambda x:x[1], reverse= True)

selected_variables = ['diabetes','admission_age','chronic_pulmonary','TEMPC','LVEF','MEAN_BP','RedBloodCells','RR','eth_WHITE'] #1

#select the best k and the variables
best_k = 11
best_k=np.argmax(results_for_each_k)+kmin
selected_variables=vars_for_each_k[best_k]

#plot silhouette score
trains = tr_n[selected_variables].copy()
sil_score = []
for k in range(3,15):
    kmedoids = KMedoids(n_clusters=k, **kmed_kwargs).fit(trains)
    train_clusters = kmedoids.predict(trains)
    sil = silhouette_score(trains, train_clusters, metric= 'manhattan')
    sil_score.append(sil)
sil_score = pd.DataFrame(sil_score, index = range(3,15), columns=['silhouette score'])


rel = sns.relplot(x=tsne[1],
                y=tsne[0],
                #style = 'tt',
                hue = 'cluster',
                data = tsne,
                palette = 'deep',
                col = 'set',
                )
rel.fig.suptitle('K-medoids clustering')


###implement k-medoids with 10-fold CV

kf = KFold(n_splits=10, shuffle = True, random_state=1996)

#for k in range(3,10):
    k=6
    rmseij = []
    maeij = []
    for train_index, test_index in kf.split(tr_n[selected_variables]):
        X_train , X_test = tr_n.iloc[train_index,:],tr_n.iloc[test_index,:]
        kmedoids = KMedoids(n_clusters=k, **kmed_kwargs).fit(X_train[selected_variables])
        train_clusters=kmedoids.predict(X_train[selected_variables])
        train_clusters = pd.DataFrame(train_clusters, index = X_train.index, columns = ['cluster']) #check index
        trainx = pd.merge(train_clusters, pivot_class, left_index= True, right_index = True)
        trainx_mean = trainx.groupby(['cluster']).mean()
        test_clusters=kmedoids.predict(X_test[selected_variables])
        test_clusters = pd.DataFrame(test_clusters,index = X_test.index, columns = ['cluster'])
        testx = pd.merge(test_clusters, pivot_class, left_index= True, right_index = True)
        testx_mean = testx.groupby(['cluster']).mean()
        subtract = testx_mean.subtract(trainx_mean)
        sqr = np.square(subtract)
        div = abs(subtract).sum(axis = 1)
        rmseij.append(np.sqrt(sqr.sum(axis = 1)/18))    
        maeij.append(div/18)
    rmseij = pd.DataFrame(rmseij)
    maeij = pd.DataFrame(maeij)
    rmse = (rmseij.sum(axis = 0)/10).sum()/k
    mae = (maeij.sum(axis = 0)/10).sum()/k
    #print('selected features are',selected_variables)
    #print('number of cluster =',k,',rmse =',rmse, ',MAE =',mae)
    kmedoids = KMedoids(n_clusters=k, **kmed_kwargs).fit(tr_n[selected_variables])
    tsne_model = TSNE(n_components= 2, verbose= 1, random_state= 1990, n_iter = 500)
    tsne = tsne_model.fit_transform(tr_n[selected_variables])
    tsne = pd.DataFrame(tsne)
    tsne['k'] = kmedoids.predict(tr_n[selected_variables])
    for cluster in np.unique(kmedoids.labels_): # plot data by cluster
        plt.scatter(x=tsne.where(tsne['k']==cluster)[1],
                    y=tsne.where(tsne['k']==cluster)[0],
                    label=cluster)
    plt.legend(fontsize = 7);
    plt.title(selected_variables)
    plt.suptitle(['number of cluster =',k,',rmse =',rmse, ',MAE =',mae],fontsize = 10, va = 'baseline')
    plt.show()


#training with k-medoids
trains = tr_n[selected_variables].copy()
test_n = scaler.fit_transform(test)
test_n = pd.DataFrame(test_n, columns = test.columns, index = test.index)
tests = test_n[selected_variables].copy()
kmedoids = KMedoids(n_clusters=6, **kmed_kwargs).fit(trains)
train_clusters=kmedoids.predict(trains)
train_clusters = pd.DataFrame(train_clusters, index = trains.index, columns = ['cluster'])
trainx = pd.merge(train_clusters, pivot_class, left_index= True, right_index = True)
trainx_mean = trainx.groupby(['cluster']).mean()
test_clusters=kmedoids.predict(tests)
test_clusters = pd.DataFrame(test_clusters,index = tests.index, columns = ['cluster'])
testx = pd.merge(test_clusters, pivot_class, left_index= True, right_index = True)
testx_mean = testx.groupby(['cluster']).mean()

tsne_model = TSNE(n_components= 2, random_state= 0, n_iter = 500)
full = pd.concat([trains,tests])
tsne = tsne_model.fit_transform(full)

tsne = pd.DataFrame(tsne,index = full.index)
tsne['cluster'] = train_clusters['cluster']
tsne['cluster'].iloc[946:1352]  = test_clusters['cluster']
tsne['set']= 'test'
tsne['set'].iloc[0:946] = 'train'

tsne1 = pd.DataFrame(tsne)
tsne1['cluster'] = train_clusters['cluster']
tsne1['cluster'].iloc[0:946] = train_clusters['cluster']
tsne1['cluster'].iloc[946:1352]  = test_clusters['cluster']
tsne1['set']= 'test'
tsne1['set'].iloc[0:946] = 'train'

tsne['cluster'] = tsne['cluster'].replace([0,1,2,3,4,5],[1,2,3,4,5,6]).astype(int)
#tsne['cluster'] = tsne['cluster'].replace([0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10]).astype(int)
tsne1['cluster'] = tsne1['cluster'].replace([0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10]).astype(int)

sns.set()
fig, axes=plt.subplots(1,3, figsize=(15, 5))
sns.lineplot(data = sil_score,
             marker = 'o',
            ax = axes[0]).set(xlabel='Number of clusters, k', title = 'Silhouette score for K-medoids clustering')
sns.scatterplot(x=tsne[1], y = tsne[0], hue = 'cluster', data=tsne[tsne['set']=='train'], ax = axes[1], legend = False,palette= 'Paired').set(title = 'k=6',xlabel = 'silhouette score = 0.295',ylabel=None)
sns.scatterplot(x=tsne1[1], y = tsne1[0], hue = 'cluster', data=tsne1[tsne1['set']=='train'], ax = axes[2],palette = 'Paired').set(title = 'k=9',xlabel = 'silhouette score = 0.310',ylabel = None, yticklabels=[])
axes[2].legend(title = 'Clusters',loc = 'lower left',fontsize=10,bbox_to_anchor=(1, 0.15))#,labels = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10']

rel = sns.relplot(x=tsne[1],
                y=tsne[0],
                #style = 'tt',
                hue = 'cluster',
                data = tsne,
                palette = 'Paired',
                col = 'set',
                ).set(xlabel='',ylabel='')
trainx2022 = trainx.copy()
trainx1 = trainx.copy()
#rel.fig.suptitle('K-medoids clustering')

#set a threshold of 0.5
threshold=tx_perc/100
comparison = trainx_mean.gt(threshold, axis=1)
traintx = trainx_mean.mask(~comparison, 0)
#traintx= traintx.mask(traintx != 1, 0)
testtx = testx_mean.mask(~comparison, 0)
#testtx= testtx.mask(testtx != 1, 0)

rmseadj =[]
maeadj = []
subtract = testtx.subtract(traintx)
sqr = np.square(subtract)
div = abs(subtract).sum(axis = 1)
rmseadj.append(np.sqrt(sqr.sum(axis = 1)/18))    
maeadj.append(div/18)
rmseadj = pd.DataFrame(rmseadj)
maeadj = pd.DataFrame(maeadj)


##
##
##DBSCAN based with gower distance

categorical_features = list(catvar.columns)
features = list(tr_n.columns)
cat = [True if x in categorical_features else False for x in selected_variables]
gd = gower.gower_matrix(tr_n[selected_variables],cat_features = cat)

eps = 0.111
minsample = 8
db = DBSCAN(eps = eps, min_samples= minsample, metric = 'precomputed').fit(gd)
labels = db.labels_

tsne_model = TSNE(n_components= 2, random_state= 1999, n_iter = 500)
tsne = tsne_model.fit_transform(tr_n[selected_variables])
tsne = pd.DataFrame(tsne)
tsne['k'] = labels

for cluster in np.unique(labels): # plot data by cluster
    plt.scatter(x=tsne.where(tsne['k']==cluster)[1],
                y=tsne.where(tsne['k']==cluster)[0],
                label=cluster)
plt.legend(prop={'size': 6});
plt.suptitle(['eps =', eps,'min_sample = ', minsample])

##select the constant pool
vars_count = {}
vars_list = sum(vars_for_each_k.values(), [])
for i in np.unique(vars_list):
    vars_count[i] = vars_list.count(i)
sorted(vars_count.items(), key=lambda x:x[1], reverse= True)

def predict(db, test):
    dists = test.iloc[:,db.core_sample_indices_]
    new_clusters = []
    for x in range(0,len(test)):
        i=np.argmin(dists.iloc[x])
        new_clusters.append(db.labels_[db.core_sample_indices_[i]] if dists.iloc[x,i] < db.eps else -1 )
    return new_clusters


kf = KFold(n_splits=10, shuffle = True, random_state=1996)
#for minsample in range(5,11):
    rmseij = []
    maeij = []
    for train_index, test_index in kf.split(tr_n):
        cat = [True if x in categorical_features else False for x in selected_variables]
        newdata = tr_n[selected_variables].copy()
        gd = gower.gower_matrix(newdata,cat_features = cat)
        gd = pd.DataFrame(gd)
        X_train , X_test = gd.iloc[train_index,train_index],gd.iloc[test_index,:]
        db = DBSCAN(eps = 0.119, min_samples= 5, metric = 'precomputed').fit(X_train)
        labels = db.labels_
        #new indexing from 0 to len(tr_n)
        newdata['nindex']=range(0,len(tr_n))
        tx_class = pd.merge(newdata['nindex'],pivot_class,left_index= True, right_index = True)
        tx_class = tx_class.set_index('nindex')        
        train_clusters = pd.DataFrame(labels, index = X_train.index, columns = ['cluster']) #check index
        trainx = pd.merge(train_clusters, tx_class, left_index= True, right_index = True)
        trainx_mean = trainx.groupby(['cluster']).mean()
        #need to remove -1
        trainx_mean = trainx_mean[(trainx_mean.index != -1)]
        test_clusters = predict(db,X_test)
        test_clusters = pd.DataFrame(test_clusters,index = X_test.index, columns = ['cluster'])
        testx = pd.merge(test_clusters, tx_class, left_index= True, right_index = True)
        testx_mean = testx.groupby(['cluster']).mean()
        testx_mean = testx_mean[(testx_mean.index !=  -1)]
        subtract = testx_mean.subtract(trainx_mean)
        sqr = np.square(subtract)
        div = abs(subtract).sum(axis = 1)
        rmseij.append(np.sqrt(sqr.sum(axis = 1)/18))    
        maeij.append(div/18)
    rmseij = pd.DataFrame(rmseij)
    maeij = pd.DataFrame(maeij)
    rmse = (rmseij.sum(axis = 0)/10).sum()/8
    rmse
    mae = (maeij.sum(axis = 0)/10).sum()/8
    print('selected features are',selected_variables)
    print(',rmse =',rmse, ',MAE =',mae)
    db = DBSCAN(eps = eps, min_samples= minsample, metric = 'precomputed').fit(gd)
    tsne_model = TSNE(n_components= 2, verbose= 1, random_state= 1990, n_iter = 500)
    tsne = tsne_model.fit_transform(tr_n[selected_variables])
    tsne = pd.DataFrame(tsne)
    tsne['k'] = db.labels_
    for cluster in np.unique(db.labels_): # plot data by cluster
        plt.scatter(x=tsne.where(tsne['k']==cluster)[1],
                    y=tsne.where(tsne['k']==cluster)[0],
                    label=cluster)
    plt.legend(prop={'size': 6});
    plt.title(selected_variables)
    plt.suptitle('rmse = %s, MAE = %s, minsample = %s, eps = %s' % (rmse,mae,minsample,eps),fontsize = 10, va = 'baseline')
    plt.show()

#compare treatment between test and training
def newindex(data,treatment):
    data['nindex'] = range(0,len(data))
    tx_class = pd.merge(data['nindex'],treatment,left_index= True, right_index = True)
    tx_class = tx_class.set_index('nindex')
    return tx_class       
    

trains = tr_n[selected_variables].copy()
test_n = scaler.fit_transform(test)
test_n = pd.DataFrame(test_n, columns = test.columns, index = test.index)
tests = test_n[selected_variables].copy()
dist = pd.concat([trains,tests])
cat = [True if x in categorical_features else False for x in selected_variables]
gd = gower.gower_matrix(dist,cat_features = cat)
gd = pd.DataFrame(gd)

db = DBSCAN(eps = 0.11, min_samples= 5, metric = 'precomputed').fit(gd.iloc[0:946,0:946]) #1: 0.12; 1000:
tx_class = newindex(dist,pivot_class)
labels = db.labels_
train_clusters = pd.DataFrame(labels, columns = ['cluster']) #check index
trainx = pd.merge(train_clusters, tx_class, left_index= True, right_index = True)
trainx_mean = trainx.groupby(['cluster']).mean()
trainx_mean = trainx_mean[(trainx_mean.index != -1)]
test_in = gd.iloc[946:1352,:]
test_clusters = predict(db,test_in)
test_clusters = pd.DataFrame(test_clusters,index = test_in.index, columns = ['cluster'])
testx = pd.merge(test_clusters, tx_class, left_index= True, right_index = True)
testx_mean = testx.groupby(['cluster']).mean()
testx_mean = testx_mean[(testx_mean.index !=  -1)]

tsne_model = TSNE(n_components= 2, verbose= 1, random_state= 0, n_iter = 500)
full = pd.concat([trains,tests])
tsne = tsne_model.fit_transform(full)


tsne = pd.DataFrame(tsne)
tsne['cluster'] = train_clusters['cluster']
tsne['cluster'].iloc[946:1352]  = test_clusters['cluster']
tsne['cluster'] = tsne['cluster'].replace([0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8])
tsne['cluster'] = tsne['cluster'].astype(int)
tsne['cluster'] = tsne['cluster'].replace([-1],'noise')
tsne['set']= 'test'
tsne['set'].iloc[0:946] = 'train'
tsne[tsne['cluster'] == 'noise']


tsne1 = pd.DataFrame(tsne)
tsne1['cluster'] = train_clusters['cluster']
tsne1['cluster'].iloc[946:1352]  = test_clusters['cluster']
tsne1['cluster'] = tsne1['cluster'].replace([0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8]).astype(int)
tsne1['cluster'] = tsne1['cluster'].replace([-1],'noise')
tsne1['set']= 'test'
tsne1['set'].iloc[0:946] = 'train'


sns.set()
fig, axes = plt.subplots(1,2,figsize=(15, 5))
sns.scatterplot(x=tsne[1], y = tsne[0], hue = 'cluster', data=tsne[tsne['set']=='train'], legend = False, ax = axes[1], palette = 'Paired').set(title = 'eps = 0.12',xlabel = 'silhouette score = 0.433',ylabel=None)
sns.scatterplot(x=tsne1[1], y = tsne1[0], hue = 'cluster', data=tsne1[tsne1['set']=='train'], ax = axes[0], palette = 'Paired').set(title = 'eps = 0.11',xlabel = 'silhouette score = 0.425',ylabel = None, yticklabels=[])#
axes[0].legend(title='cluster',loc = 'lower left',fontsize=10,bbox_to_anchor=(2.25, 0.15))
               

tsne['cluster'] = tsne['cluster'].replace([0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8]).astype(int)
rel = sns.relplot(x=tsne[1],
                y=tsne[0],
                col = 'set',
                hue = 'cluster',
                data = tsne,
                palette = 'Paired'
                ).set(xlabel = '',ylabel = '')
rel.fig.suptitle('')#DBSCAN clustering

#threshold 0.5
#trainx_sum = trainx.groupby(['cluster']).sum()
#trainx_av = trainx_sum.div(trainx_sum.sum(axis=0),axis=1)

threshold=tx_perc/100
comparison = trainx_mean.gt(threshold, axis=1)
traintx = trainx_mean.mask(~comparison, 0)
#traintx= traintx.mask(traintx != 1, 0)
testtx = testx_mean.mask(~comparison, 0)

rmseadj =[]
maeadj = []
subtract = testtx.subtract(traintx)
sqr = np.square(subtract)
div = abs(subtract).sum(axis = 1)
rmseadj.append(np.sqrt(sqr.sum(axis = 1)/18))    
maeadj.append(div/18)
rmseadj = pd.DataFrame(rmseadj)
maeadj = pd.DataFrame(maeadj)

silhouette_score(gd.iloc[0:946,0:946], labels ,metric = 'precomputed') #0.3600252
