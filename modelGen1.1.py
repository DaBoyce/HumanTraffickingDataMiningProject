import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

import warnings
warnings.filterwarnings("ignore")

##################################################################################################################################
###### IMPORTANT VARIABLES #######################################################################################################
##################################################################################################################################
inputFileName = 'human_trafficking.csv'

us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

targets = ['meansOfControlDebtBondage', 'meansOfControlTakesEarnings', 'meansOfControlRestrictsFinancialAccess', 
           'meansOfControlThreats', 'meansOfControlPsychologicalAbuse', 'meansOfControlPhysicalAbuse', 
           'meansOfControlSexualAbuse', 'meansOfControlFalsePromises', 'meansOfControlPsychoactiveSubstances', 
           'meansOfControlRestrictsMovement', 'meansOfControlRestrictsMedicalCare', 'meansOfControlExcessiveWorkingHours', 
           'meansOfControlUsesChildren', 'meansOfControlThreatOfLawEnforcement', 'meansOfControlWithholdsNecessities', 
           'meansOfControlWithholdsDocuments', 'meansOfControlOther', 'meansOfControlNotSpecified', 
           'isForcedLabour', 'isSexualExploit', 'isOtherExploit', 'isSexAndLabour', 'isForcedMarriage', 'isForcedMilitary', 
           'isOrganRemoval', 'isSlaveryAndPractices', 'typeOfLabourAgriculture', 
           'typeOfLabourAquafarming', 'typeOfLabourBegging', 'typeOfLabourConstruction', 'typeOfLabourDomesticWork', 
           'typeOfLabourHospitality', 'typeOfLabourIllicitActivities', 'typeOfLabourManufacturing', 'typeOfLabourMiningOrDrilling', 
           'typeOfLabourPeddling', 'typeOfLabourTransportation', 'typeOfLabourOther', 'typeOfLabourNotSpecified', 'typeOfSexProstitution', 'typeOfSexPornography', 'typeOfSexRemoteInteractiveServices', 
           'typeOfSexPrivateSexualServices', 'recruiterRelationIntimatePartner', 'recruiterRelationFriend', 
           'recruiterRelationFamily', 'recruiterRelationOther', 'recruiterRelationUnknown', 'isAbduction']

##################################################################################################################################
###### CLEAN DATA ################################################################################################################
##################################################################################################################################
dataFrame = pd.read_csv(inputFileName)

# Replace -99 (missing value placeholder) with NaN
dataFrame.replace(-99, np.nan, inplace=True)

# Modify citizenship values: if they are a two-letter abbreviation of any US state, replace them with 'US'
dataFrame['citizenship'] = dataFrame['citizenship'].apply(lambda x: 'US' if x in us_states else x)

# Handle missing values
for col in dataFrame.columns:
    if dataFrame[col].dtype == 'object':
        dataFrame[col].fillna(dataFrame[col].mode()[0], inplace=True)
    else:
        dataFrame[col].fillna(dataFrame[col].median(), inplace=True)

# Convert gender to binary (1 for Female, 0 for Male)
dataFrame['gender'] = dataFrame['gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Convert categorical variables to dummy variables
df = pd.get_dummies(dataFrame, columns=['ageBroad', 'citizenship'])

# Update features to include newly created dummy columns
features = ['gender']
features.extend([col for col in df.columns if 'ageBroad_' in col])
features.extend([col for col in df.columns if 'citizenship_' in col])

# Convert target variable to binary and handle missing values
for target in targets:
    if df[target].dtype == 'object':
        df[target] = df[target].apply(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else np.nan)
        df[target].fillna(df[target].median(), inplace=True)

# Prepare the features and targets
X = df[features]
y = df[targets]

# Handle missing values in the features
X = X.fillna(X.median())

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize empty list to hold models and their scores
model_data = []

# Define class weights to iterate over
class_weights = ['balanced']

##################################################################################################################################
###### MODEL CREATION ############################################################################################################
##################################################################################################################################
for i in range(101):
    n = i / 100
    class_weights.append({0: 1 - n, 1: n})

if os.path.isfile('best_model_weight_index.txt'):
    print("Best Model Exists")
    # Load the weights of the best model
    with open('best_model_weight_index.txt', 'r') as f:
        best_model_index = int(f.read())
else:
    print("Beginning Model Testing")
    best_model_index = 0
    best_f1 = 0

    # Loop through class_weights
    for i, weight in enumerate(class_weights):
        # Create and train the model
        clf = MultiOutputRegressor(RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weight))
        clf.fit(X_train_scaled, y_train)

        # Make predictions and calculate f1 score
        y_pred = clf.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Append model, score, and iteration number to the list
        model_data.append((clf, f1))

        # Check if current model's f1 is better than the best f1 so far
        if f1 > best_f1:
            best_f1 = f1
            best_model_index = i

        print(f"Model {i} Completed. F1: {f1}")

    # Print the results of the best model
    best_model, best_score = model_data[best_model_index]
    print(f"Best model is model {best_model_index} with f1 score of {best_score}")
    print(classification_report(y_test, best_model.predict(X_test_scaled), target_names=targets))

    # Save the weights of the best model
    with open('best_model_weight_index.txt', 'w') as f:
        f.write(str(best_model_index))

##################################################################################################################################
###### MODEL RETRAINING ##########################################################################################################
##################################################################################################################################
print(f"Retraining Model")

# Retrain the best model on the entire dataset
X_scaled = scaler.fit_transform(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = MultiOutputRegressor(RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights[best_model_index]))
clf.fit(X_train_scaled, y_train)

# Make predictions and calculate f1 score
y_pred = clf.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results of the best model
best_model = clf
best_score = f1 

print(f"Model Completed")


##################################################################################################################################
##### DATA OUTPUT ################################################################################################################
##################################################################################################################################
# Relavent Features
# Confusion Matrix
# ROC Curves
# Heat Map
# Precision Recall Curves

##################################################################################################################################
##### RELAVENT FEATURES ##########################################################################################################
##################################################################################################################################
# Feature importance
importances = best_model.estimators_[0].feature_importances_
indices = np.argsort(importances)

# Set a threshold for feature importance
threshold = 0.01

relevant_features = [features[i] for i in indices if importances[i] > threshold]
plt.figure(figsize=(10,15))
plt.title('Feature Importances')
plt.barh(range(len(relevant_features)), [importances[i] for i in indices if importances[i] > threshold], color='b', align='center')
plt.yticks(range(len(relevant_features)), relevant_features)
plt.xlabel('Relative Importance')
plt.savefig('feature_importances.png')
plt.close()
print("Relevant Features Complete")

##################################################################################################################################
##### CONFUSION MATRIX ##########################################################################################################
##################################################################################################################################
for i, target in enumerate(targets):
    cm = confusion_matrix(y_test[target], y_pred[:, i])
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix for ' + target)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_{target}.png')
    plt.close()
print("Confusion Matrices Complete")

##################################################################################################################################
##### ROC CURVES #################################################################################################################
##################################################################################################################################
for i, target in enumerate(targets):
    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test[target], y_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Create a new figure
    plt.figure()

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for ' + target)
    plt.legend(loc="lower right")

    # Save the ROC curve to a file
    plt.savefig(f'roc_curve_{target}.png')

    # Close the figure to free up memory
    plt.close()
print("ROC Curves Complete")

##################################################################################################################################
##### HEAT MAP ###################################################################################################################
##################################################################################################################################
all_columns = features + targets

# Select only columns of interest
selected_data = df[all_columns]

# Calculate correlation matrix
correlation_matrix = selected_data.corr()

# Create the correlation heatmap
plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix, cmap="YlGnBu")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
print("Heatmap Complete")

##################################################################################################################################
##### PRECISION-RECALL CURVES ####################################################################################################
##################################################################################################################################
# Iterate over each target class
for i, target in enumerate(targets):
    # Calculate Precision-Recall curve and AP score
    precision, recall, _ = precision_recall_curve(y_test[target], y_pred[:, i])
    average_precision = average_precision_score(y_test[target], y_pred[:, i])

    # Create a new figure
    plt.figure()

    # Plot the Precision-Recall curve
    plt.step(recall, precision, where='post', label=f'Average Precision={average_precision:0.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve for {target}')

    # Save the Precision-Recall curve to a file
    plt.savefig(f'precision_recall_{target}.png')

    # Close the figure to free up memory
    plt.close()
print("Precision-Recall Curves Complete")
