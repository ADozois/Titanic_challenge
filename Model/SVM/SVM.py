import pickle
import xml.etree.ElementTree
import sklearn.svm as SVMSK
from tensorflow.contrib.learn import SVM
from abc import ABCMeta, abstractmethod


class Svm:
    __metaclass__ = ABCMeta

    def __init__(self, xml_config):
        self.core = self.__build_model(xml_config)
        self.prediction = None

    @property
    def core(self):
        return self.core

    @core.setter
    def core(self, value):
        raise Exception("You can't directly assign a core to the model")

    @property
    def prediction(self):
        return self.prediction

    @prediction.setter
    def prediction(self, value):
        raise Exception("You can't directly assign a prediction to the model")

    @abstractmethod
    def __build_model(self, xml_file):
        """Should build the model of SVM"""
        return

    @abstractmethod
    def fit(self, train_data):
        """Should fit the best line to separate data"""
        return

    @abstractmethod
    def predict(self, predict_data):
        """For predicting the class of the input data"""
        return

    @abstractmethod
    def __parse_xml(self, xml_config):
        """Parse configuration for model
        :param **kwargs:
        """
        return

    def save(self, file_name, file_path):
        if file_path:
            file_name = file_path + file_name
        pickle.dump(self.core, open(file_name, "wb"))

    def load(self, file_name, file_path):
        if file_path:
            file_name = file_path + file_name
        self.core = pickle.load(open(file_name, "rb"))


class SvmTf(Svm):
    def __init__(self, config_file):
        super().__init__(config_file)

    def __build_model(self, xml_file):
        svm_params = self.__parse_xml(xml_file)
        self.core = SVM(example_id_column=svm_params['exemple_id_column'], feature_columns=svm_params['feature_column'],
                        weight_column_name=svm_params['weight_column_name'], model_dir=svm_params['model_dir'],
                        l1_regularization=svm_params['l1_regularization'],
                        l2_regularization=svm_params['l2_regularization'],
                        num_loss_partitions=svm_params['num_loss_partitions'], kernels=svm_params['kernels'],
                        config=svm_params['config'], feature_engineering_fn=svm_params['feature_engineering_fn'])

    def __parse_xml(self, xml_config):
        params = {}
        xml_params = xml_config.find('params')
        for param in xml_params:
            params.update(param.attrib)
        return params

    def fit(self, train_data):
        self.core.fit(train_data)

    def predict(self, predict_data):
        self.prediction = self.core.predict(predict_data)
        return self.prediction


class SvmSk(Svm):
    def __init__(self, xml_file):
        super().__init__(xml_file)

    def __build_model(self, xml_file):
        svm_params = self.__parse_xml(xml_file)
        if xml_file.get('type') == 'linear':
            self.core = SVMSK.LinearSVC(penalty=svm_params['penalty'], loss=svm_params['loss'], dual=svm_params['dual'],
                                        tol=svm_params['tol'], C=svm_params['C'], multi_class=svm_params['multi_class'],
                                        fit_intercept=svm_params['fit_intercept'],
                                        intercept_scaling=svm_params['intercept_scaling'],
                                        class_weight=svm_params['class_weight'], verbose=svm_params['verbose'],
                                        random_state=svm_params['random_state'], max_iter=svm_params['max_itr'])
        else:
            self.core = SVMSK.SVR(kernel=svm_params['kernel'], degree=svm_params['degree'], gamma=svm_params['gamma'],
                                  coef0=svm_params['coef0'], tol=svm_params['tol'], C=svm_params['C'],
                                  epsilon=svm_params['epsilon'], shrinking=svm_params['shrinking'],
                                  cache_size=svm_params['cache_size'], verbose=svm_params['verbose'],
                                  max_iter=svm_params['max_iter'])

    def __parse_xml(self, xml_config):
        params = {}
        if xml_config.get('type') == 'linear':
            xml_params = xml_config.find('params_lin')
        else:
            xml_params = xml_config.find('params_rbf')
        for param in xml_params:
            params.update(param.attrib)
        return params

    def fit(self, train_data):
        self.core.fit(train_data[0],train_data[1])

    def predict(self, predict_data):
        self.core.predict(predict_data)


class SvmFactory:
    @staticmethod
    def create_svm(file):
        xml_file = xml.etree.ElementTree.parse(file).getroot()
        if xml_file.get('backend') == 'tf':
            svm = SvmTf(xml_file)
        else:
            svm = SvmSk(xml_file)
        return svm
