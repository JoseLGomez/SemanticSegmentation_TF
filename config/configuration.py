import imp
import os


class Configuration():
    def __init__(self, config_path, exp_name, exp_folder):
        self.config_path = config_path
        self.exp_name = exp_name
        self.exp_folder = exp_folder + exp_name + '/'
        if not os.path.exists(self.exp_folder):
    		os.makedirs(self.exp_folder)

    def Load(self):
        cf = imp.load_source('config', self.config_path)
        cf.config_path = self.config_path
        cf.exp_name = self.exp_name
        cf.exp_folder = self.exp_folder
        if cf.predict_output is None:
            cf.predict_output = self.exp_folder + 'predictions/'
            if not os.path.exists(cf.predict_output):
                os.makedirs(cf.predict_output)
        if cf.resize_image_train is not None:
            cf.size_image_train = cf.resize_image_train
        if cf.resize_image_valid is not None:
            cf.size_image_valid = cf.resize_image_valid 
        if cf.resize_image_test is not None:
            cf.size_image_test = cf.resize_image_test
        if cf.model_path is None:
            cf.model_path = cf.exp_folder          
        return cf