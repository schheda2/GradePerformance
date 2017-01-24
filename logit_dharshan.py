#------------------------------------Author-Punniya dharshan-------------------------------------------------------

import pandas as pd
import numpy
df=pd.read_csv("E:/UNCC/Fall 16/Machine Learning/project/Source_Files/Final/Variable_reduced_data_by_PCA.csv")
print df.columns
x=pd.DataFrame()
y=pd.DataFrame()
x['Student_1_5_Grade_performance']=df['Student_1_5_Grade_performance']
x['Schools_with_high_no_of_low_income_families']=df['Schools_with_high_no_of_low_income_families']
x['Probabaility that 2 student belong to different Race']=df['Probabaility that 2 student belong to different Race']
x['Class Size Index']=df['Class Size Index']
x['Experience of teacher']=df['Experience of teacher']
x['No of Expulsion per 100 student']=df['No of Expulsion per 100 student']
x['Long Suspensions per 100 students']=df['Long Suspensions per 100 students']
y['Grade_binary']=df['Grade_binary']


#part of code to spli train and test data
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=51)

#part of code to run Logit 
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(penalty='l2',C=1)
model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
print "Logistics Accuracy is %2.2f" % accuracy_score(y_test,model.predict(x_test))

#from sklearn.metrics import roc_auc_curve
from sklearn.metrics import classification_report
#logit_roc_auc=roc_auc_curve(y_test,model.predict(x_test))
#print "Logistic ROC AUC is %2.2f" % logit_roc_auc


print classification_report(y_test,model.predict(x_test))


#code to print the number of values in each class
#print df.groupby('Grade_binary').count()

#ROC Curve

#from sklearn.metrics import roc_curve
#fpr,tpr,thresholds=roc_curve(y_test,model.predict_proba(x_test)[:,1]) 

#part of code that prints summary of Logit
import statsmodels.api as sm
cols_to_keep = ['Student_1_5_Grade_performance', 'Schools_with_high_no_of_low_income_families', 'Probabaility that 2 student belong to different Race','Class Size Index','Experience of teacher','No of Expulsion per 100 student','Long Suspensions per 100 students']
data = df[cols_to_keep]
 

logit = sm.Logit(df['Grade_binary'], data[cols_to_keep])
result = logit.fit()
print result.summary()

#part of code to claculate TP,TP,FP,FN

pred=model.predict(x_test)
pred=numpy.array(pred)


TP=0
TN=0
FP=0
FN=0
w=0
li1=list()
test_act=list()
for test in y_test['Grade_binary']:
    test_act.append(test)
test_act=numpy.array(test_act)

while w<test_act.size:
    li1.append(w)
    w=w+1
print "-----------------------------------------------------------------------"
for i in li1:
    if test_act[i]==1:
        if test_act[i]==pred[i]:
            TP=TP+1
        else:
            FN=FN+1
    else:
        if test_act[i]==pred[i]:
            TN=TN+1
        else:
            FP=FP+1
print "Confusion Matrix"
print "-----------------------------------------------------------------------------"
print "TP TN"
print TP,TN
print "FP FN"
print FP,FN
print "----------------------"
#part of code to get count of 0 and 1 in test data set.
size=test_act.size
li=list()
v=0
c0=0
c1=0

data=[]
for line in y_test['Grade_binary']:
    data.append(line)
data=numpy.array(data)

while v<size:
    li.append(v)
    v=v+1
for i in li:
    if data[i]==0:
        c0=c0+1
    else:
        c1=c1+1
print "Total no of actual 0 are %2d" %c0
print "Total no of actual 1 are %2d" %c1


