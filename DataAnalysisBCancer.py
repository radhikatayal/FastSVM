import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#Data=pd.read_csv("pima-indians-diabetes.csv")
Data=pd.read_csv("breast_cancer_heart_disease_data2.csv")

print(Data.describe())

Data.hist(figsize=(10,8))

plt.show()

Corr=Data[Data.columns].corr()
#print(Corr)
sns.heatmap(Corr,annot=True)
plt.show()
#print(Data.BMI.value_counts())
#print(Data.SkinThickness.value_counts())
#print(Data.Insulin.value_counts())
##sns.set()
##sns.pairplot(Data, vars = ["BMI","DiabetesPedigreeFunction","Age"])

##plt.show()
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# How varaibles are distributed
##g = sns.PairGrid(Data, vars=['Glucose', 'Insulin', 'BMI'], hue="Outcome", size=2.4)
##g.map_diag(plt.hist)
##g.map_upper(plt.scatter)
##g.map_lower(sns.kdeplot, cmap="Blues_d")
##g.add_legend()

plt.show()

##g = sns.PairGrid(Data, vars=['Age', 'SkinThickness', 'BloodPressure'], hue="Outcome", size=2.4)
##g.map_diag(plt.hist)
##g.map_upper(plt.scatter)
##g.map_lower(sns.kdeplot, cmap="Blues_d")
##g.add_legend()
#plt.show()

##g = sns.PairGrid(Data, vars=['Pregnancies', 'DiabetesPedigreeFunction'], hue="Outcome", size=3.5)
##g.map_diag(plt.hist)
##g.map_upper(plt.scatter)
##g.map_lower(sns.kdeplot, cmap="Blues_d")
##g.add_legend()
#plt.show()

# vairable corelation
columns = ['Glucose', 'Age', 'BloodPressure', 'Insulin','BMI','SkinThickness' ,'Pregnancies',  'DiabetesPedigreeFunction']
n_cols = 2
n_rows = 4
idx = 0

##for i in range(n_rows):
##    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(8, 2.4))
##    for j in range(n_cols):
##        sns.violinplot(x = Data.Outcome, y=Data[columns[idx]], ax=ax[j])
##        idx += 1
##        if idx >= 8:
##            break
plt.show()
#sns.pointplot()


X = Data.iloc[:,0:10]
Y = Data.iloc[:,10]

#missing values
for i in range(0,10):
    colunm = Data.iloc[:,i]
    print("feature",Data.columns[i],": ", len(colunm[colunm==0]))


#model = DecisionTreeClassifier(max_depth=4,random_state=0)
#model.fit(X, Y)

def plot_feature_importances_adult_census(model):
    n_features = Data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),Data.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("feature")
    ##plt.show()
    ##fig=plt.figure()
    plt.savefig("feature_imporatnace_diabetes.png")
    plt.show()
    plt.close()
#plot_feature_importances_adult_census(model)

#From the above data exploration, we saw an outlier of SkinThickness
# remove the Outlier of skin thickness

#max_skinthickness = Data.SkinThickness.max()
#data = Data[Data.SkinThickness!=max_skinthickness]
data = Data
#Replace zero value of Glucose, BloodPressure, SkinThickness, Insulin, BMI with mean for each class
def replace_zero(df, field, target):
    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
    data.loc[(df[field] == 0)&(df[target] == 0), field] = mean_by_target.iloc[0][0]
    data.loc[(df[field] == 0)&(df[target] == 1), field] = mean_by_target.iloc[1][0]

    # run the function
##for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
##    replace_zero(data, col, 'Outcome')



