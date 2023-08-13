import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 320)

relative_path = 'train.csv'
titanic_data = pd.read_csv(relative_path)


# Data pre-processing ; handling the missing values
print(titanic_data.isnull().sum())

# a) drop the cabin column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# b) replace the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# c) replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
print(titanic_data.isnull().sum())

# Data Analysis
# getting some statistical measures about the data
print(titanic_data.describe())

# finding the number of people survived and not survived
print(titanic_data['Survived'].value_counts())

# Data visualization
sns.set()

# making a count plot for "Survived" column
sns.countplot('Survived', data=titanic_data)
plt.show()

# number of survivors Gender wise
sns.countplot('Sex', hue='Survived', data=titanic_data)
plt.show()

# Encoding the Categorical Column
print(titanic_data['Sex'].value_counts())

print(titanic_data['Embarked'].value_counts())

# converting categorical columns

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)

# Seperate features and target columns
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']


# Splitting the data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=2)

# neural network
# build a model
model = Sequential()
model.add(Dense(16, input_shape=(X.shape[1],), activation='relu')) # Add an input shape! (features,)
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# compile the model
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# early stopping callback
# This callback will stop the training when there is no improvement in
# the validation loss for 10 consecutive epochs.
es = EarlyStopping(monitor='val_accuracy',
                                   mode='max', # don't minimize the accuracy!
                                   patience=10,
                                   restore_best_weights=True)

# now we just update our model fit call
history = model.fit(X,
                    Y,
                    callbacks=[es],
                    epochs=80, # you can set this to a big number!
                    batch_size=10,
                    validation_split=0.2,
                    shuffle=True,
                    verbose=1)

history_dict = history.history
# Learning curve(Loss)
# let's see the training and validation loss by epoch

# loss
loss_values = history_dict['loss'] # you can change this
val_loss_values = history_dict['val_loss'] # you can also change this

# range of X (no. of epochs)
epochs = range(1, len(loss_values) + 1)

# plot
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Learning curve(accuracy)
# let's see the training and validation accuracy by epoch

# accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# orange is for "orange"
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# this is the max value - should correspond to
# the HIGHEST train accuracy
np.max(val_acc)

# Learning curve(accuracy)
# let's see the training and validation accuracy by epoch

# accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# orange is for "orange"
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# this is the max value - should correspond to
# the HIGHEST train accuracy
np.max(val_acc)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# see how these are numbers between 0 and 1?
model.predict(X) # prob of successes (survival)
np.round(model.predict(X),0) # 1 and 0 (survival or not)
Y # 1 and 0 (survival or not)

# so we need to round to a whole number (0 or 1),
# or the confusion matrix won't work!
preds = np.round(model.predict(X),0)

# confusion matrix
print(confusion_matrix(Y, preds)) # order matters! (actual, predicted)
print(classification_report(Y, preds))




# Model training: Logistic Regression model
log_reg = LogisticRegression()
# training the logistic regression model with training data
log_reg.fit(X_train, Y_train)
log_predictions = log_reg.predict(X_train)
print(log_predictions)
log_train_accuracy = accuracy_score(Y_train, log_predictions)
print(log_train_accuracy)

# accuracy on test data
log_test_prediction = log_reg.predict(X_test)
print(log_test_prediction)

log_test_accuracy = accuracy_score(Y_test, log_test_prediction)
print(log_test_accuracy)

log_mse = mean_squared_error(Y_train, log_predictions)
log_rmse = np.sqrt(log_mse)
print("LogisticRegression_rmse:", log_rmse)



# Model training: RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, Y_train)
forest_reg_prediction = forest_reg.predict(X_train)
for_mse = mean_squared_error(Y_train, forest_reg_prediction)
for_rmse = np.sqrt(for_mse)
print("RandomForestRegressor_rmse:", for_rmse)

# Model training: DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, Y_train)
tree_reg_prediction = tree_reg.predict(X_train)
tree_mse = mean_squared_error(Y_train, tree_reg_prediction)
tree_rmse = np.sqrt(tree_mse)
print("DecisionTreeRegressor_rmse:", tree_rmse)


# Correlation Matrix
corr = titanic_data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(titanic_data.corr(),annot = True)
plt.show()


from sklearn.tree import DecisionTreeClassifier        # Decision tree
clf_dt = DecisionTreeClassifier().fit(X_train,Y_train)
y_pred_dt = clf_dt.predict(X_test)

from sklearn.ensemble import RandomForestClassifier   # Random forest
clf_rf = RandomForestClassifier().fit(X_train,Y_train)
y_pred_rf = clf_rf.predict(X_test)

from sklearn.svm import SVC                            # Support Vector Machine
clf_svm = SVC(gamma=1, C=1000,probability=True).fit(X_train,Y_train)
y_pred_svm = clf_svm.predict(X_test)

from sklearn.naive_bayes import GaussianNB, BernoulliNB    # Naive Bayes (Bernoulli)
clf_nb = BernoulliNB().fit(X_train,Y_train)
y_pred_nb = clf_nb.predict(X_test)

from sklearn.ensemble import AdaBoostClassifier        # Ada boost
clf_ada = AdaBoostClassifier(n_estimators=100, random_state =0).fit(X_train, Y_train)
y_pred_ada = clf_ada.predict(X_test)

from sklearn.neural_network import MLPClassifier        # Multi-layer Perceptron
clf_mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(1000,), random_state=1).fit(X_train, Y_train)
y_pred_mlp = clf_mlp.predict(X_test)

# accuracy_score
from sklearn.metrics import accuracy_score
acc_dt = accuracy_score(Y_test, y_pred_dt)
acc_rf = accuracy_score(Y_test, y_pred_rf)
acc_svm = accuracy_score(Y_test, y_pred_svm)
acc_nb = accuracy_score(Y_test, y_pred_nb)
acc_ada = accuracy_score(Y_test, y_pred_ada)
acc_mlp = accuracy_score(Y_test, y_pred_mlp)
acc_log = accuracy_score(Y_test, log_test_prediction)

print("The accuracy of Decision Tree: {} %".format(acc_dt*100))
print("The accuracy of Random Forest: {} %".format(acc_rf*100))
print("The accuracy of Support Vector Mchine: {} %".format(acc_svm*100))
print("The accuracy of Bernoulli Naive Bayes: {} %".format(acc_nb*100))
print("The accuracy of Ada Boost: {} %".format(acc_ada*100))
print("The accuracy of Multi-layer Perceptron: {} %".format(acc_mlp*100))
print("The accuracy of logistic regression: {} %".format(acc_log*100))
print(" ")

# precision_score
from sklearn.metrics import precision_score
prec_dt = precision_score(Y_test,y_pred_dt,average='weighted')
prec_nb = precision_score(Y_test,y_pred_nb,average='weighted')
prec_rf = precision_score(Y_test,y_pred_rf,average='weighted')
prec_svm = precision_score(Y_test,y_pred_svm,average='weighted')
prec_ada = precision_score(Y_test,y_pred_ada,average='weighted')
prec_mlp = precision_score(Y_test,y_pred_mlp,average='weighted')
prec_log = precision_score(Y_test,log_test_prediction,average='weighted')


print("The precision of Decision Tree: {} %".format(prec_dt*100))
print("The precision of Bernoulli Naive Bayes: {} %".format(prec_nb*100))
print("The precision of SVM: {} %".format(prec_svm*100))
print("The precision of Random Forest: {} %".format(prec_rf*100))
print("The precision of Ada Boost: {} %".format(prec_ada*100))
print("The precision of Multi-layer Perceptron: {} %".format(prec_mlp*100))
print("The precision of logistic regression: {} %".format(prec_log*100))
print(" ")

# recall score
from sklearn.metrics import recall_score
recall_dt = recall_score(Y_test,y_pred_dt,average='weighted')
recall_nb = recall_score(Y_test,y_pred_nb,average='weighted')
recall_rf = recall_score(Y_test,y_pred_rf,average='weighted')
recall_svm = recall_score(Y_test,y_pred_svm,average='weighted')
recall_ada = recall_score(Y_test,y_pred_ada,average='weighted')
recall_mlp = recall_score(Y_test,y_pred_mlp,average='weighted')
recall_log = recall_score(Y_test,log_test_prediction,average='weighted')

print("The recall of Decision Tree: {} %".format(recall_dt*100))
print("The recall of Bernoulli Naive Bayes: {} %".format(recall_nb*100))
print("The recall of SVM: {} %".format(recall_svm*100))
print("The recall of Random Forest: {} %".format(recall_rf*100))
print("The recall of Ada Boost: {} %".format(recall_ada*100))
print("The recall of Multi-layer Perceptron: {} %".format(recall_mlp*100))
print("The recall of logistic regression: {} %".format(recall_log*100))
print(" ")

# f1 score
from sklearn.metrics import f1_score
f1_dt = f1_score(y_pred_dt,Y_test,average='weighted')
f1_nb = f1_score(y_pred_nb,Y_test,average='weighted')
f1_svm = f1_score(y_pred_svm,Y_test,average='weighted')
f1_rf = f1_score(y_pred_rf,Y_test,average='weighted')
f1_ada = f1_score(y_pred_ada,Y_test,average='weighted')
f1_mlp = f1_score(y_pred_mlp,Y_test,average='weighted')
f1_log = f1_score(log_test_prediction,Y_test,average='weighted')

print("The F1-score of Decision Tree: {} %".format(f1_dt*100))
print("The F1-score of Bernoulli Naive Bayes: {} %".format(f1_nb*100))
print("The F1-score of SVM: {} %".format(f1_svm*100))
print("The F1-score of Random Forest: {} %".format(f1_rf*100))
print("The F1-score of Ada Boost: {} %".format(f1_ada*100))
print("The F1-score of Multi-layer Perceptron: {} %".format(f1_mlp*100))
print("The F1-score of logistic regression: {} %".format(f1_log*100))
print(" ")


from sklearn.model_selection import cross_val_score, cross_val_predict
dt_scores = cross_val_predict(clf_dt, X_train, Y_train, cv=3)
nb_scores = cross_val_predict(clf_nb, X_train, Y_train, cv=3)
svm_scores = cross_val_predict(clf_svm, X_train, Y_train, cv=3)
rf_scores = cross_val_predict(clf_rf, X_train, Y_train, cv=3)
ada_scores = cross_val_predict(clf_ada, X_train, Y_train, cv=3)
mlp_scores = cross_val_predict(clf_mlp, X_train, Y_train, cv=3)
log_scores = cross_val_predict(log_reg, X_train, Y_train, cv=3)


from sklearn.metrics import precision_recall_curve
dt_precisions, dt_recalls, dt_thresholds = precision_recall_curve(Y_train, dt_scores)
nb_precisions, nb_recalls, nb_thresholds = precision_recall_curve(Y_train, nb_scores)
svm_precisions, svm_recalls, svm_thresholds = precision_recall_curve(Y_train, svm_scores)
rf_precisions, rf_recalls, rf_thresholds = precision_recall_curve(Y_train, rf_scores)
ada_precisions, ada_recalls, ada_thresholds = precision_recall_curve(Y_train, ada_scores)
mlp_precisions, mlp_recalls, mlp_thresholds = precision_recall_curve(Y_train, mlp_scores)
log_precisions, log_recalls, log_thresholds = precision_recall_curve(Y_train, log_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


# roc_curve
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics

dt_fpr, dt_tpr, dt_thresholds = roc_curve(Y_train, dt_scores)
nb_fpr, nb_tpr, nb_thresholds = roc_curve(Y_train, nb_scores)
svm_fpr, svm_tpr, svm_thresholds = roc_curve(Y_train, svm_scores)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(Y_train, rf_scores)
ada_fpr, ada_tpr, ada_thresholds = roc_curve(Y_train, ada_scores)
mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(Y_train, mlp_scores)
log_fpr, log_tpr, log_thresholds = roc_curve(Y_train, log_scores)

def plot_roc_curve(fpr,tpr, label=None):
    plt.plot(fpr,tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC for different classifier")
    plt.legend(loc="lower right")

roc_auc_rf = metrics.auc(rf_fpr,rf_tpr)
roc_auc_svm = metrics.auc(svm_fpr,svm_tpr)
roc_auc_ada = metrics.auc(ada_fpr,ada_tpr)
roc_auc_mlp = metrics.auc(mlp_fpr,mlp_tpr)
roc_auc_nb = metrics.auc(nb_fpr,nb_tpr)
roc_auc_log = metrics.auc(log_fpr,log_tpr)
roc_auc_dt = metrics.auc(dt_fpr,dt_tpr)

plt.plot(rf_fpr, rf_tpr, color="red", linestyle=":",label="Random Forest (area = %0.2f)" % roc_auc_rf)
plt.plot(svm_fpr, svm_tpr, color="green", linestyle=":",label="Support vector machine (area = %0.2f)" % roc_auc_svm)
plt.plot(ada_fpr, ada_tpr, color="pink", linestyle=":",label="Ada boost (area = %0.2f)" % roc_auc_ada)
plt.plot(mlp_fpr, mlp_tpr, color="aqua", linestyle=":",label="Multi-layer Perceptron (area = %0.2f)" % roc_auc_mlp)
plt.plot(nb_fpr, nb_tpr, color="orange", linestyle=":",label="Naive Bayes (Bernoulli) (area = %0.2f)" % roc_auc_nb)
plt.plot(log_fpr, log_tpr, color="yellow", linestyle=":",label="Logistic Regression (area = %0.2f)" % roc_auc_log)
plot_roc_curve(dt_fpr, dt_tpr ,label="Decision tree (area = %0.2f)" % roc_auc_dt)
plt.show()


