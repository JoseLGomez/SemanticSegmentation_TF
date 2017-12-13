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
        return cf