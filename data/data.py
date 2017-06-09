import pandas as pd
import pickle
import math


class Data:
    def __init__(self):
        self.data = None
        self.size = (0, 0)

    def load_data_from_file(self, filename, list_stand, type_norm='s', list_feature=None):
        pre_data = pd.read_csv(filename)
        pre_data = self.__id_as_index(pre_data)
        pre_data = self.__preprocess(pre_data)
        pre_data = self.__clean_data(pre_data,['Age'])
        if list_feature:
            pre_data = self.__extract_column(pre_data, list_feature)
        if type_norm == 's':
            pre_data = self.__standardization(pre_data,list_stand)
        elif type_norm == 'n':
            pre_data = self.__min_max(pre_data, list_stand)
        self.data = pre_data
        self.size = pre_data.shape()



    @staticmethod
    def __extract_column(data, list_feature):
        return data.loc[:, list_feature]

    @staticmethod
    def __preprocess(data):
        data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'female' else 0)
        data['Embarked'] = data['Embarked'].map(lambda x: 0 if x == 'C' else x)
        data['Embarked'] = data['Embarked'].map(lambda x: 1 if x == 'Q' else x)
        data['Embarked'] = data['Embarked'].map(lambda x: 2 if x == 'S' else x)
        return data

    @staticmethod
    def __clean_data(data, list_to_mean):
        for feature in list(data):
            if feature in list_to_mean:
                data[feature] = data[feature].map(lambda x: data[feature].mean() if math.isnan(x) else x)
            else:
                data[feature] = data[feature].map(lambda x: 0 if math.isnan(x) else x)
            return data

    @staticmethod
    def __standardization(data, list_to_standardize):
        for feature in list_to_standardize:
            data[feature] = data[feature].map(lambda x: (x - data[feature].mean()) / data[feature].std())
        return data

    @staticmethod
    def __min_max(data, list_to_normalize):
        for feature in list_to_normalize:
            data[feature] = data[feature].map(
                lambda x: (x - data[feature].min()) / (data[feature].max() - data[feature].min()))
            return data

    @staticmethod
    def __id_as_index(data):
        data = data.set_index(data['PassengerId'])
        del data['PassengerId']
        return data

    def save_data(self, name=None, file_path=None):
        if not name:
            name = 'titanic_data.p'
        if file_path:
            name = file_path + name
        pickle.dump(self.data, open(name, 'wb'))

    def load_dataset(self, name=None, file_path=None):
        if not name:
            name = 'titanic_data.p'
        if file_path:
            name = file_path + name
        self.data = pickle.load(open(name, 'rb'))
