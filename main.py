# Machine Learning - An Application to the "Churn" Problem
# coding: utf-8

# Load libraries
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Convert categorical values to numeric
def categorical2numeric(df):
    """
    Converts categorical values to numeric
    :param df: Trainset or Testset
    :return: Pandas dataframe
    """
    df['international_plan'] = df.apply(lambda x: 1 if x['international_plan']=='yes'
                                                   else 0, axis=1)

    df['voice_mail_plan'] = df.apply(lambda x: 1 if x['voice_mail_plan']=='yes'
                                                   else 0, axis=1)

    df['area_code'] = df.apply(lambda x: 1 if x['area_code']=='area_code_415'
                                                    else 2 if x['area_code']=='area_code_408'
                                                   else 3, axis=1)

    df['state'] = df.groupby(['state']).ngroup()

    return df


# Create new features
def new_features(df):
    """
    Creates new features based on the initial features
    :param df: Trainset or Testset
    :return: Pandas dataframe
    """
    # create new feature 'total_minutes'
    df['total_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes']
    # create new feature 'total_calls'
    df['total_calls'] =  df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls']
    # create new feature 'total_charge'
    df['total_charge'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge']
    # create new feature 'mins_per_call'
    df['mins_per_call'] = df['total_minutes'] / df['total_calls']
    # create new feature 'charge_per_call'
    df['charge_per_call'] = df['total_charge'] / df['total_calls']
    # create new feature 'mins_per_call_intl'
    df['mins_per_call_intl'] = df['total_intl_minutes'] / df['total_intl_calls']
    # create new feature 'charge_per_call_intl'
    df['charge_per_call_intl'] = df['total_intl_charge'] / df['total_intl_calls']
    # create new feature 'when_more_calls'
    df['when_more_calls'] = df.apply(lambda x: 1 if (x['total_day_calls']>=x['total_eve_calls']) & (x['total_day_calls']>=x['total_night_calls'])
                                                    else 2 if (x['total_eve_calls']>=x['total_day_calls']) & (x['total_eve_calls']>=x['total_night_calls'])
                                                   else 3 if (x['total_night_calls']>=x['total_day_calls']) & (x['total_night_calls']>=x['total_eve_calls'])
                                                 else 4
                                                 ,axis=1)
    # change null values with zeros
    df['charge_per_call_intl'].fillna(0, inplace=True)
    df['mins_per_call_intl'].fillna(0, inplace=True)

    return  df

# scale features
def scale_features(df, to_scale):
    """
    Scales features to [-1,1]
    :param df:  Trainset or Testset
    :param to_scale: List with the names of the features that need to be scaled
    :return: Pandas dataframe
    """
    scaler = preprocessing.StandardScaler()
    df[to_scale] = scaler.fit_transform(df[to_scale])

    return df


def resampling(df_train, over=True):
    """
    
    :param df_train: The trainset
    :param over: True means that oversampling will be performed. Else undersampling.
    :return: Pandas dataframe
    """
    # Class count
    count_class_0, count_class_1 = df_train.churn.value_counts()

    # Divide by class
    df_class_0 = df_train[df_train['churn'] == 'no']
    df_class_1 = df_train[df_train['churn'] == 'yes']

    if over == True:
        # Over sample
        df_class_1_over = df_class_1.sample(count_class_0, replace=True)
        df2 = pd.concat([df_class_0, df_class_1_over], axis=0)
    else:
        # Under sample
        df_class_0_under = df_class_0.sample(count_class_1)
        df2 = pd.concat([df_class_0_under, df_class_1], axis=0)

    return df2


def conf_matrix(df_train, features):
    """
    Creates and shows a confusion matrix of the XGB classifier
    :param df_train: The trainset
    :param features: A list with the features of the trainset
    :return: Nothing
    """
    X = df_train[features]
    Y = df_train['churn']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    clf = XGBClassifier(max_depth=6).fit(x_train,y_train)
    #clf.predict(x_train)
    y_pred = clf.predict(x_test)

    # Creates a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index=['no', 'yes'],
                         columns=['no', 'yes'])

    plt.figure(figsize=(5.5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Random Forest \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main():
    """
    This is the main function of the proccess. 
    It reads, preprocesses the data. 
    Trains the model.
    Uses the model to make predictions.
    Creates the final csv file that should be submitted to kaggle
    :return: Nothing
    """
    # Read the data
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    print('- Go to the train set')

    # Preprocessing the train set

    print('-- Convert categorical features to numeric')
    df_train = categorical2numeric(df_train)

    print('-- Create new features based on the initial data set.')
    df_train = new_features(df_train)
    
    # delete the following highly correlated feautures - Not necessary
    # del df_train['total_day_minutes']
    # del df_train['total_eve_minutes']
    # del df_train['total_night_minutes']
    # del df_train['total_day_calls']
    # del df_train['total_eve_calls']
    # del df_train['total_night_calls']
    # del df_train['total_day_charge']
    # del df_train['total_eve_charge']
    # del df_train['total_night_charge']

    # scale features
    #print('-- Scale dataset')
    #df_train = scale_features(df_train, to_scale)

    # perform resampling on the train set
    #print('Resampling')
    #df_train = resampling(df_train, over=True)

    # Create a list of the feature column's names
    df_train_features = df_train.copy()
    del df_train_features['churn']
    features = df_train_features.columns

    # Train the model
    print('-- Train the model')

    # create a list of the target(churn) values
    y = pd.factorize(df_train['churn'])[0]

    # Create a XGboost Classifier.
    #clf = RandomForestClassifier(n_jobs=2, criterion="entropy")
    clf = XGBClassifier(max_depth=6)

    # Train the Classifier to take the training features and learn how they relate
    # to the training y (churn)
    clf.fit(df_train[features], y)

    # create a confusion matrix
    #print('-- Creating a confusion matrix')
    #conf_matrix(df_train, features)

    # - Test set
    print('- Go to the test set')

    # Preprocessing the test set

    print('-- Convert categorical features to numeric')
    df_test = categorical2numeric(df_test)

    print('-- Create new features based on the initial data set.')
    df_test = new_features(df_test)

    # scale features
    #print('-- Scale dataset')
    #df_test = scale_features(df_test, to_scale)

    # create a list with the index (1-750)
    ids = df_test.id.tolist()

    print('-- Make predictions')
    # Apply the Classifier we trained to the test data
    predictions = clf.predict(df_test[features])

    print('- Create the final csv file that will be submitted to kaggle')

    # create a dataframe in the kaggle submition format
    df_submit = pd.DataFrame({'churn':predictions, 'id':ids})
    # set column id as index
    df_submit = df_submit.set_index('id')

    # convert churn from [0,1] to ['no','yes']
    df_submit['churn'] = df_submit.apply(lambda x: 'yes' if x['churn']==1
                                                   else 'no' if x['churn']==0
                                                   else 3, axis=1)

    # check the percentage of the target in the predictions
    print(df_submit['churn'].value_counts(normalize=True) * 100)

    # write the predictions to a csv file
    # this is the csv file that will be submitted to kaggle
    df_submit.to_csv('predictions.csv')

    print('End of process.')

if __name__ == '__main__' :
    main()