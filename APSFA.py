
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import weka.core.serialization as serialization
from weka.classifiers import FilteredClassifier
from sklearn.metrics import mean_squared_error
from sklearn.impute import IterativeImputer
import weka.core.converters as converters
from sklearn.impute import SimpleImputer
from weka.classifiers import Evaluation
from weka.classifiers import Classifier
from sklearn.decomposition import PCA
import weka.core.packages as packages
import weka.plot.classifiers as plcls
from weka.core.classes import Random
from sklearn.metrics import r2_score
from prettytable import PrettyTable
from xgboost import XGBRegressor
from weka.filters import Filter
import matplotlib.pyplot as plt
import weka.core.jvm as jvm
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
# %matplotlib inline
import warnings



warnings.filterwarnings('ignore')


def printfeatures(coef,features):
    new_features = []
    for i in range(len(coef)):
        if not (coef[i] == 0):
            new_features.append(features[i])
    print(len(new_features),"features are selected. They are [",new_features[0],","
          ,new_features[1],",",new_features[2],",",new_features[3],"...",new_features[len(new_features)-1],"]")


# Cloning Git Repo
! git clone https://github.com/adityasamant/APS-Failure-Analysis

# Reading File Names from .names file
with open("communities.names","r") as namefile:
    attributes = []
    for line in namefile:
        if line.startswith("@attribute"):
            words = line.split()
            attributes.append(words[1])

# Reading Data from .data file
data = pd.read_csv('communities.data', header=None, na_values = '?', names=attributes)

# Train Test Split
train = data.iloc[:1495]
test = data.iloc[1495:]
print("The size of Train Data is",len(train),"and size of Test Data is",len(test))

# Showing Data
data.head(5)



IMP = SimpleImputer()
trainX = pd.DataFrame(IMP.fit_transform(train.iloc[:,5:-1]),columns=attributes[5:-1])
testX = pd.DataFrame(IMP.transform(test.iloc[:,5:-1]),columns=attributes[5:-1])

trainy = train['ViolentCrimesPerPop']
testy = test['ViolentCrimesPerPop']

print(trainX.shape)
trainX.head(5)

print(testX.shape)
testX.head(5)


f, ax = plt.subplots(figsize=(20, 18))
corr = trainX.corr()
ax = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10 ,as_cmap=True),
            square=True, ax=ax, linewidth=.005)



CoV = trainX.std()/trainX.mean()
CoV[:5]



# root 128 is 11
topcov = CoV.sort_values(ascending=False)[:11].axes[0].to_list()

i = 0
def diagfunc(x, **kws):
    global i
    ax = plt.gca()
    ax.annotate(topcov[i], xy=(0.5, 0.5), xycoords=ax.transAxes, horizontalalignment='center',
            verticalalignment='center')
    i = i+1

# Using seaborn pairgrid to create scatterplot matrix
scatplot = sns.PairGrid(trainX,palette=["#072960","#ff0004"],vars=topcov)
scatplot = scatplot.map_offdiag(plt.scatter,edgecolor="k",s=40)
scatplot = scatplot.map_diag(diagfunc)
scatplot = scatplot.add_legend()

for axis in scatplot.axes.flatten():
    axis.set_xlabel("")
    axis.set_ylabel("")



plt.figure(figsize=(13, 10))
ax = sns.boxplot(data=trainX[topcov],palette=["#072960"],orient='h')



LRmodel1f = LinearRegression(n_jobs=-1).fit(trainX,trainy)
predict = LRmodel1f.predict(testX)
print('The test error is',mean_squared_error(testy, predict))


besta = []
a = 0.00001
while a<10000:
    LRmodel1g = Ridge(alpha=a)
    
    # Running Cross Validation
    scores = cross_val_score(LRmodel1g,trainX,trainy,cv=5,n_jobs=-1)   
    besta.append(scores.mean())
    
    a = a * 10

bestA = 0.00001 * 10**besta.index(max(besta))
LRmodel1g = Ridge(alpha=bestA)
LRmodel1g.fit(trainX,trainy)
print("Test Error for best \u03BB =",bestA,"is",1-LRmodel1g.score(testX,testy))



besta = []
a = 0.00001
while a<10003:
    LRmodel1h = Lasso(alpha=a)
    
    # Running Cross Validation
    scores = cross_val_score(LRmodel1h,trainX,trainy,cv=5,n_jobs=-1)   
    besta.append(scores.mean())
    
    a = a * 10

bestA = 0.00001 * 10**besta.index(max(besta))
LRmodel1h = Lasso(alpha=bestA)
LRmodel1h.fit(trainX,trainy)
print("Test Error with best \u03BB =",bestA,"is",1-LRmodel1h.score(testX,testy))
printfeatures(LRmodel1h.coef_.tolist(),trainX.columns)

sc = StandardScaler()
NtrainX = sc.fit_transform(trainX)
NtestX = sc.transform(testX)

besta1 = []
a = 0.00001
while a<10003:
    LRmodel1h1 = Lasso(alpha=a)
    
    # Running Cross Validation
    scores = cross_val_score(LRmodel1h1,NtrainX,trainy,cv=5,n_jobs=-1)   
    besta1.append(scores.mean())
    
    a = a * 10

bestA1 = 0.00001 * 10**besta1.index(max(besta1))
LRmodel1h1 = Lasso(alpha=bestA1)
LRmodel1h1.fit(NtrainX,trainy)
print("\nTest Error for Standardized features with best \u03BB =",bestA1,"is",1-LRmodel1h1.score(NtestX,testy))
printfeatures(LRmodel1h1.coef_.tolist(),trainX.columns)



bestm = []
for m in range(1,len(trainX.columns)+1):
    pcrmodel1i = PCA(n_components=m)
    trainX_pca = pcrmodel1i.fit_transform(trainX)
    
    LRmodel1i = LinearRegression(n_jobs=-1)
    
    scores = cross_val_score(LRmodel1i,trainX_pca,trainy,cv=5,n_jobs=-1)   
    bestm.append(scores.mean())
    
bestM = bestm.index(max(bestm))+1

pcrmodel1i = PCA(n_components=bestM)
trainX_pca = pcrmodel1i.fit_transform(trainX)
testX_pca = pcrmodel1i.transform(testX)
LRmodel1i = LinearRegression(n_jobs=-1).fit(trainX_pca,trainy)
print("Test Error with best m =",bestM,"is",1-LRmodel1i.score(testX_pca,testy))



bestalpha = []
a = 0.00001
while a<10000:
    xgb = XGBRegressor(reg_alpha=a, reg_lambda=0,n_estimators=100,n_jobs=-1)
    
    # Running Cross Validation
    scores = cross_val_score(xgb,trainX,trainy,cv=5,n_jobs=-1)   
    bestalpha.append(scores.mean())
    
    a = a * 10

bestAlpha = 0.00001 * 10**bestalpha.index(max(bestalpha))
xgb = XGBRegressor(reg_alpha=bestAlpha, reg_lambda=0,n_estimators=100,n_jobs=-1)
xgb.fit(trainX,trainy)
pred = xgb.predict(testX)
print("Test Error with best \u03B1 =",bestAlpha,"is",1-r2_score(testy,pred))



train2 = pd.read_csv('aps_failure_training_set.csv', na_values = 'na',skiprows=20)
test2 = pd.read_csv('aps_failure_test_set.csv', na_values = 'na',skiprows=20)

print(train2.shape)
train2.head(5)

print(test2.shape)
test2.head(5)


if(Path('test.csv').exists() and Path('train.csv').exists()):
    train2 = pd.read_csv('train.csv')
    test2 = pd.read_csv('test.csv')
    train2y = train2['class']
    train2X = train2.drop(['class'],axis = 1)
    test2y = test2['class']
    test2X = test2.drop(['class'],axis = 1)
else:
    IMP = IterativeImputer()
    train2X = pd.DataFrame(IMP.fit_transform(train2.iloc[:,1:]),columns=train2.columns[1:])
    test2X = pd.DataFrame(IMP.transform(test2.iloc[:,1:]),columns=test2.columns[1:])
    train2y = train2['class']
    test2y = test2['class']
    train2 = train2X.copy()
    train2['class'] = train2y
    test2 = test2X.copy()
    test2['class'] = test2y
    train2.to_csv(path_or_buf="train.csv",index= False)
    test2.to_csv(path_or_buf="test.csv",index= False)

print(train2X.shape)
train2X.head(5)

print(test2X.shape)
test2X.head(5)



CoV2 = train2X.std()/train2X.mean()
CoV2[165:170]


f, ax = plt.subplots(figsize=(20, 18))
corr = train2X.corr()
ax = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10 ,as_cmap=True),
            square=True, ax=ax, linewidth=.005)


# root 170 is 13
topcov2 = CoV2.sort_values(ascending=False)[:13].axes[0].to_list()

temp = []
for name in topcov2:
    temp.append("")
    temp.append(name)

i = 0
def diagfunc(x, **kws):
    global i
    ax = plt.gca()
    ax.annotate(temp[i], xy=(0.5, 0.5), xycoords=ax.transAxes, horizontalalignment='center',
            verticalalignment='center')
    i = i+1

# Using seaborn pairgrid to create scatterplot matrix
scatplot = sns.PairGrid(train2, hue='class',palette=["#072960","#ff0004"],vars=topcov2)
scatplot = scatplot.map_offdiag(plt.scatter,edgecolor="k",s=40)
scatplot = scatplot.map_diag(diagfunc)
scatplot = scatplot.add_legend()

for axis in scatplot.axes.flatten():
    axis.set_xlabel("")
    axis.set_ylabel("")

# Setting Size for Box Plots and adjusting width between them
plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=1)

# Creating Each Boxplot Recursively
for axis in range(len(topcov2)):
    plt.subplot(2,7,axis+1)
    sns.boxplot(x="class",y=topcov2[axis],palette=["#104599","Red"],data=train2)



print("Positive data:",test2y.tolist().count("pos")+train2y.tolist().count("pos"),"\tNegative data:",test2y.tolist().count("neg")+train2y.tolist().count("neg"))
print("Yes, the data is imbalaced.")



rfc = RandomForestClassifier(oob_score=True,n_jobs = -1, n_estimators=100).fit(train2X,train2y)
pred = rfc.predict(train2X)
predt = rfc.predict(test2X)
cm = confusion_matrix(train2y,pred)
cmt = confusion_matrix(test2y,predt)

# Converting Confusion Matrix to Data Frame and Plotting
cmD = pd.DataFrame(cmt, index = ["neg","pos"],columns = ["neg","pos"])
plt.figure(figsize = (7,7))
axis = sns.heatmap(cmD, annot=True, cbar=False, cmap="Reds")


# plot ROC Curve
fpr, tpr, threshold = roc_curve(np.where(test2y=='neg', 0, 1),np.where(predt=='neg', 0, 1))
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.03, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


print("Misclassifications for train set are",cm[0][1]+cm[1][0])
print("Misclassifications for test set are",cmt[0][1]+cmt[1][0])
print("\nOOB Error is",1 - rfc.oob_score_)
print("Test Error is",1 - rfc.score(test2X,test2y))


rfc1 = RandomForestClassifier(oob_score=True,class_weight="balanced", n_jobs = -1,n_estimators=100).fit(train2X,train2y)
pred1 = rfc1.predict(train2X)
predt1 = rfc1.predict(test2X)
cm1 = confusion_matrix(train2y,pred1)
cmt1 = confusion_matrix(test2y,predt1)

# Converting Confusion Matrix to Data Frame and Plotting
cmD1 = pd.DataFrame(cmt1, index = ["neg","pos"],columns = ["neg","pos"])
plt.figure(figsize = (7,7))
axis = sns.heatmap(cmD1, annot=True, cbar=False, cmap="Reds")


# plot ROC Curve
fpr, tpr, threshold = roc_curve(np.where(test2y=='neg', 0, 1),np.where(predt1=='neg', 0, 1))
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.03, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


print("Misclassifications for train set are",cm1[0][1]+cm1[1][0])
print("Misclassifications for test set are",cmt1[0][1]+cmt1[1][0])
print("\nOOB Error is",1 - rfc1.oob_score_)
print("Test Error is",1 - rfc1.score(test2X,test2y))


t = PrettyTable(["Method","Train Score","Test Score","OOB Score"])
t.add_row(["Random Forest",rfc.score(train2X,train2y),rfc.score(test2X,test2y),rfc.oob_score_])
t.add_row(['RF with compensation',rfc1.score(train2X,train2y),rfc1.score(test2X,test2y),rfc1.oob_score_])
print(t)



jvm.start(max_heap_size="4g",packages=True)


Wtrain = converters.load_any_file("train.csv")
Wtest = converters.load_any_file("test.csv")
Wtrain.class_is_last()
Wtest.class_is_last()



if(Path('lmt.model').exists()):
    lmt = Classifier(jobject=serialization.read("lmt.model"))
else:
    lmt = Classifier(classname="weka.classifiers.trees.LMT")
    lmt.build_classifier(Wtrain)
    serialization.write("lmt.model", lmt)

evlmt = Evaluation(Wtrain)
evlmt.crossvalidate_model(lmt, Wtrain, 5, Random(1))



print("Error is",evlmt.error_rate)
cm2e = evlmt.confusion_matrix
cm2E = pd.DataFrame(cm2e, index = ["neg","pos"],columns = ["neg","pos"])
plt.figure(figsize = (7,7))
axis = sns.heatmap(cm2E, annot=True, cbar=False, cmap="Reds")
plcls.plot_roc(evlmt,class_index=[1])


tevlmt = Evaluation(Wtrain)
tevlmt.test_model(lmt,Wtest)


print("Error is",tevlmt.error_rate)
tcm2e = tevlmt.confusion_matrix
tcm2E = pd.DataFrame(tcm2e, index = ["neg","pos"],columns = ["neg","pos"])
plt.figure(figsize = (7,7))
axis = sns.heatmap(tcm2E, annot=True, cbar=False, cmap="Reds")
plcls.plot_roc(tevlmt,class_index=[1])


packages.install_package("SMOTE")


smote = Filter(classname="weka.filters.supervised.instance.SMOTE",options=["-P", "4800"])
smt = Classifier(classname="weka.classifiers.trees.LMT")
fc = FilteredClassifier()
fc.filter = smote
fc.classifier = smt
fc.build_classifier(Wtrain)


evsmt = Evaluation(Wtrain)
evsmt.crossvalidate_model(fc, Wtrain, 5, Random(1))

print("Error is",evsmt.error_rate)
cm2f = evsmt.confusion_matrix
cm2F = pd.DataFrame(cm2f, index = ["neg","pos"],columns = ["neg","pos"])
plt.figure(figsize = (7,7))
axis = sns.heatmap(cm2F, annot=True, cbar=False, cmap="Reds")
plcls.plot_roc(evsmt,class_index=[1])


tevsmt = Evaluation(Wtrain)
tevsmt.test_model(fc,Wtest)


print("Error is",tevsmt.error_rate)
tcm2f = tevsmt.confusion_matrix
tcm2F = pd.DataFrame(tcm2f, index = ["neg","pos"],columns = ["neg","pos"])
plt.figure(figsize = (7,7))
axis = sns.heatmap(tcm2F, annot=True, cbar=False, cmap="Reds")
plcls.plot_roc(tevsmt,class_index=[1])


t = PrettyTable(["Method","Train Error","Test Error"])
t.add_row(["LMT",evlmt.error_rate,tevlmt.error_rate])
t.add_row(['LMT (SMOTE)',evsmt.error_rate,tevsmt.error_rate])
print(t)


jvm.stop()
