import numpy as np 
import pandas as pd

df = pd.read_csv("credit_card.csv")

def SplitTrainSet(data, testRatio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    testSetSize = int(len(data)*testRatio)
    testIndices = shuffled[:testSetSize]
    trainIndices = shuffled[testSetSize:]
    return data.iloc[trainIndices], data.iloc[testIndices]

train_data, test_data = SplitTrainSet(df, 0.2)

class NaiveBayesClassifier:

    def __init__(self):
        pass
    
    def separate_classes(self, X, y):
        separated_classes = {}
        for i in range(len(X)):
            feature_values = X[i]
            class_name = y[i]
            if class_name not in separated_classes:
                separated_classes[class_name] = []
            separated_classes[class_name].append(feature_values)
        return separated_classes
    def stat_info(self, X):
        for feature in zip(*X):
            yield {
                'std' : np.std(feature),
                'mean' : np.mean(feature)
            }

    def distribution(self, x, mean, std):

        exponent = np.exp(-((x-mean)**2 / (2*std**2)))

        return exponent / (np.sqrt(2*np.pi)*std)

    def fit (self, X, y):
        separated_classes = self.separate_classes(X, y)
        self.class_summary = {}

        for class_name, feature_values in separated_classes.items():
            self.class_summary[class_name] = {
                'prior_proba': len(feature_values)/len(X),
                'summary': [i for i in self.stat_info(feature_values)],
            }
        return self.class_summary

    def predict(self, X):
        MAPs = []

        for row in X:
            joint_proba = {}
            
            for class_name, features in self.class_summary.items():
                total_features =  len(features['summary'])
                likelihood = 1

                for idx in range(total_features):
                    feature = row[idx]
                    mean = features['summary'][idx]['mean']
                    stdev = features['summary'][idx]['std']
                    normal_proba = self.distribution(feature, mean, stdev)
                    likelihood *= normal_proba
                prior_proba = features['prior_proba']
                joint_proba[class_name] = prior_proba * likelihood

            MAP = max(joint_proba, key= joint_proba.get)
            MAPs.append(MAP)

        return MAPs

    def accuracy(self, y_test, y_pred):
        true_true = 0

        for y_t, y_p in zip(y_test, y_pred):
            if y_t == y_p:
                true_true += 1 
        return true_true / len(y_test)


x_train = np.array(train_data[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])
y_train = np.array(train_data['Class'])
test_x = np.array(test_data[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])
y_test = np.array(test_data['Class'])

model = NaiveBayesClassifier()
model.fit(x_train,y_train)
predict_y = model.predict(test_x)
print(model.accuracy(y_test,predict_y))
