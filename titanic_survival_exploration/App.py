import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def filter_data(data, condition):
    """
    Remove elements that do not match the condition provided.
    Takes a data list as input and returns a filtered list.
    Conditions should be a list of strings of the following format:
      '<field> <op> <value>'
    where the following operations are valid: >, <, >=, <=, ==, !=

    Example: ["Sex == 'male'", 'Age < 18']
    """

    field, op, value = condition.split(" ")

    # convert value into number or strip excess quotes if string
    try:
        value = float(value)
    except:
        value = value.strip("\'\"")

    # get booleans for filtering
    if op == ">":
        matches = data[field] > value
    elif op == "<":
        matches = data[field] < value
    elif op == ">=":
        matches = data[field] >= value
    elif op == "<=":
        matches = data[field] <= value
    elif op == "==":
        matches = data[field] == value
    elif op == "!=":
        matches = data[field] != value
    else:  # catch invalid operation codes
        raise Exception("Invalid comparison operator. Only >, <, >=, <=, ==, != allowed.")

    # filter data and outcomes
    data = data[matches].reset_index(drop=True)
    return data


def survival_stats(data, outcomes, key, filters=[]):
    if key not in data.columns.values:
        print "'{}' is not a feature of the Titanic data. Did you spell something wrong?".format(key)
        return False

    # Return the function before visualizing if 'Cabin' or 'Ticket'
    # is selected: too many unique categories to display
    if key == 'Cabin' or key == 'PassengerId' or key == 'Ticket':
        print "'{}' has too many unique categories to display! Try a different feature.".format(key)
        return False

    # Merge data and outcomes into single dataframe
    all_data = pd.concat([data, outcomes], axis=1)

    # Apply filters to data
    for condition in filters:
        all_data = filter_data(all_data, condition)

    # Create outcomes DataFrame
    all_data = all_data[[key, 'Survived']]

    # Create plotting figure
    plt.figure(figsize=(8, 6))

    # 'Numerical' features
    if key == 'Age' or key == 'Fare':

        # Remove NaN values from Age data
        all_data = all_data[~np.isnan(all_data[key])]

        # Divide the range of data into bins and count survival rates
        min_value = all_data[key].min()
        max_value = all_data[key].max()
        value_range = max_value - min_value

        # 'Fares' has larger range of values than 'Age' so create more bins
        if key == 'Fare':
            bins = np.arange(0, all_data['Fare'].max() + 20, 20)
        if key == 'Age':
            bins = np.arange(0, all_data['Age'].max() + 10, 10)

        # Overlay each bin's survival rates
        nonsurv_vals = all_data[all_data['Survived'] == 0][key].reset_index(drop=True)
        surv_vals = all_data[all_data['Survived'] == 1][key].reset_index(drop=True)
        plt.hist(nonsurv_vals, bins=bins, alpha=0.6,
                 color='red', label='Did not survive')
        plt.hist(surv_vals, bins=bins, alpha=0.6,
                 color='green', label='Survived')

        # Add legend to plot
        plt.xlim(0, bins.max())
        plt.legend(framealpha=0.8)

    # 'Categorical' features
    else:

        # Set the various categories
        if key == 'Pclass':
            values = np.arange(1, 4)
        if key == 'Parch' or key == 'SibSp':
            values = np.arange(0, np.max(data[key]) + 1)
        if key == 'Embarked':
            values = ['C', 'Q', 'S']
        if key == 'Sex':
            values = ['male', 'female']

        # Create DataFrame containing categories and count of each
        frame = pd.DataFrame(index=np.arange(len(values)), columns=(key, 'Survived', 'NSurvived'))
        for i, value in enumerate(values):
            frame.loc[i] = [value, \
                            len(all_data[(all_data['Survived'] == 1) & (all_data[key] == value)]), \
                            len(all_data[(all_data['Survived'] == 0) & (all_data[key] == value)])]

        # Set the width of each bar
        bar_width = 0.4

        # Display each category's survival rates
        for i in np.arange(len(frame)):
            nonsurv_bar = plt.bar(i - bar_width, frame.loc[i]['NSurvived'], width=bar_width, color='r')
            surv_bar = plt.bar(i, frame.loc[i]['Survived'], width=bar_width, color='g')

            plt.xticks(np.arange(len(frame)), values)
            plt.legend((nonsurv_bar[0], surv_bar[0]), ('Did not survive', 'Survived'), framealpha=0.8)

    # Common attributes for plot formatting
    plt.xlabel(key)
    plt.ylabel('Number of Passengers')
    plt.title('Passenger Survival Statistics With \'%s\' Feature' % (key))
    plt.show()

    # Report number of passengers with missing values
    if sum(pd.isnull(all_data[key])):
        nan_outcomes = all_data[pd.isnull(all_data[key])]['Survived']
        print "Passengers with missing '{}' values: {} ({} survived, {} did not survive)".format( \
            key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0))


def accuracy_score_r(truth, pred):
    if len(truth) == len(pred):
        return "Estimativas tem uma precisao de {:.2f}%".format((truth == pred).mean() * 100)
    else:
        return "Numero de estimativas nao bate"


def predictions_0(data):
    predictions = []
    for _, passenger in data.iterrows():
        predictions.append(0)

    return pd.Series(predictions)


def predictions_1(data):
    predictions = []
    for _, passenger in data.iterrows():

        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            predictions.append(0)

    return pd.Series(predictions)


def predictions_2(data):
    predictions = []
    for _, passenger in data.iterrows():

        if passenger['Sex'] == 'female' \
                or (passenger['Sex'] == 'male' and passenger['Age'] < 10):
            predictions.append(1)
        else:
            predictions.append(0)

    return pd.Series(predictions)


def predictions_3(data):
    predictions = []
    for _, passenger in data.iterrows():

        if passenger['Sex'] == 'female' \
                or (passenger['Sex'] == 'male' and passenger['Age'] < 10) \
                or (passenger['Sex'] == 'male' and passenger['Age'] > 10 and passenger['Pclass'] == 1 and passenger['Fare'] > 500) \
                or (passenger['Sex'] == 'male' and 10 < passenger['Age'] < 40 and passenger['Pclass'] == 1 and 20 < passenger['Fare'] < 100) \
                or (passenger['Sex'] == 'male' and passenger['Age'] > 30 and passenger['Pclass'] > 1 and 40 < passenger['Fare'] < 60):

            predictions.append(1)
        else:
            predictions.append(0)

    return pd.Series(predictions)


def test_run():
    in_file = 'titanic_data.csv'
    full_data = pd.read_csv(in_file)

    outcomes = full_data['Survived']
    data = full_data.drop('Survived', axis=1)

    predictions = pd.Series(np.ones(5, dtype=int))
    print accuracy_score_r(outcomes[:5], predictions)

    predictions = predictions_0(data)
    print accuracy_score_r(outcomes, predictions)

    predictions = predictions_1(data)
    print accuracy_score_r(outcomes, predictions)

    predictions = predictions_2(data)
    print accuracy_score_r(outcomes, predictions)

    predictions = predictions_3(data)
    print accuracy_score_r(outcomes, predictions)

    print 'Fim'


if __name__ == '__main__':
    test_run()
