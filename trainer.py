import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# load the pickled data dictionary containing the normalized landmark data and corresponding labels
data_dict = pickle.load(open('./signdata.pickle', 'rb'))

# convert the data and labels lists from the data dictionary to numpy arrays
data = np.array(data_dict['data'], dtype=object)
labels = np.asarray(data_dict['labels'])

# pad the sequences in the data array with zeros so that all samples have the same number of features
max_len = max(len(sample) for sample in data)
data_padded = np.zeros((len(data), max_len))
for i, sample in enumerate(data):
    data_padded[i, :len(sample)] = sample

# split the data and labels into training and testing sets with a test size of 20%, shuffling the data and using stratified sampling
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# create a new random forest classifier model
trainedModel = RandomForestClassifier()

# train the model using the training data and labels
trainedModel.fit(x_train, y_train)

# use the trained model to predict the labels of the test data
y_predict = trainedModel.predict(x_test)

# calculate the accuracy of the model's predictions on the test data
score = accuracy_score(y_predict, y_test)

# print the accuracy score as a percentage
print('{}% of samples were classified accurately.'.format(score * 100))

# save the trained model as a pickled file
f = open('trainedModel.p', 'wb')
pickle.dump({'trainedModel': trainedModel}, f)
f.close()
