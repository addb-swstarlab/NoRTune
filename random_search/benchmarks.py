import os
import logging
import pandas as pd
from typing import Union

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter

import envs.params as p
from envs.spark import SparkEnv
from envs.postgres import PostgresEnv

class SparkBench(SparkEnv):
    def __init__(
        self,
        workload: str = None,
        workload_size: str = None,
        alter: bool = True,
        debugging: bool = False,
    ):
        super().__init__(workload=workload, workload_size=workload_size, alter=alter, debugging=debugging)
        # self.config_path = config_path
        # csv_data = pd.read_csv(csv_path, index_col=0)
        # self.dict_data = csv_data.to_dict(orient='index')
        
        self.spark_cs = self._generate_configspace()
        
    def _generate_configspace(self) -> ConfigurationSpace:
        hyps = []        

        for k in self.dict_data.keys():
            param = self.dict_data[k]
            _type = param['type']
            match _type:
                case 'continuous':
                    hyps.append(UniformFloatHyperparameter(k, lower=param['min'], upper=param['max'], default_value=param['default'], q=0.05))
                case 'numerical':
                    hyps.append(UniformIntegerHyperparameter(k, lower=param['min'], upper=param['max'], default_value=param['default']))
                case 'binary':
                    hyps.append(CategoricalHyperparameter(k, choices=list(range(int(param['min']), int(param['max'])+1)), default_value=param['default']))
                case 'categorical':
                    hyps.append(CategoricalHyperparameter(k, choices=list(range(int(param['min']), int(param['max'])+1)), default_value=param['default']))
        
        cs = ConfigurationSpace()
        cs.add_hyperparameters(hyps)
        return cs
    
    def save_configuration_file(self, config: Union[Configuration, dict], log_flag=False):
        assert isinstance(config, (Configuration, dict)), "config must be type Configuration or Dictionary."
        
        logging.info(f"Save configuration to {self.config_path} 💨")
        if isinstance(config, Configuration):
            config = config.get_dictionary()
        
        f = open(self.config_path, 'w')

        for k in config.keys():
            _type = self.dict_data[k]['type']
            v = config[k]
            match _type:
                case 'continuous':
                    v = round(v, 2)
                    p_unit = self.dict_data[k]['unit']
                    if p_unit == p_unit:
                        v = str(v) + p_unit
                case 'numerical':
                    p_unit = self.dict_data[k]['unit']
                    if p_unit == p_unit:
                        v = str(v) + p_unit
                case 'binary':
                    _items = self.dict_data[k]['range'].split(',')
                    v = _items[v]
                case 'categorical':
                    _items = self.dict_data[k]['range'].split(',')
                    v = _items[v]
            if log_flag:
                logging.info(f'{k}={v}')
            f.writelines(f'{k}={v}\n')
    
    def random_sampling_configuration(self) -> Configuration:
        return self.spark_cs.sample_configuration()
    
    def apply_and_run_configuration(self, load=bool):
        self.apply_configuration()
        self.run_configuration(load)

class PostgresBench(PostgresEnv):
    def __init__(
        self,
        workload: str = None,
        # workload_size: str = None,
        # alter: bool = True,
        debugging: bool = False,
    ):
        super().__init__(workload=workload, debugging=debugging)
        # self.config_path = config_path
        # csv_data = pd.read_csv(csv_path, index_col=0)
        # self.dict_data = csv_data.to_dict(orient='index')
        self.workload_size = None
        self.cs = self._generate_configspace()
        
    def _generate_configspace(self) -> ConfigurationSpace:
        hyps = []        

        for k in self.dict_data.keys():
            param = self.dict_data[k]
            _type = param['type']
            match _type:
                case 'continuous':
                    # hyps.append(UniformFloatHyperparameter(k, lower=param['min'], upper=param['max'], default_value=param['default'], q=0.05))
                    hyps.append(UniformFloatHyperparameter(k, lower=param['min'], upper=param['max'], default_value=param['default']))
                case 'numerical':
                    hyps.append(UniformIntegerHyperparameter(k, lower=param['min'], upper=param['max'], default_value=param['default']))
                case 'binary':
                    hyps.append(CategoricalHyperparameter(k, choices=list(range(int(param['min']), int(param['max'])+1)), default_value=param['default']))
                case 'categorical':
                    hyps.append(CategoricalHyperparameter(k, choices=list(range(int(param['min']), int(param['max'])+1)), default_value=param['default']))
        
        cs = ConfigurationSpace()
        cs.add_hyperparameters(hyps)
        return cs
    
    def save_configuration_file(self, config: Union[Configuration, dict], log_flag=False):
        assert isinstance(config, (Configuration, dict)), "config must be type Configuration or Dictionary."
        
        logging.info(f"Save configuration to {self.config_path} 💨")
        if isinstance(config, Configuration):
            config = config.get_dictionary()
        
        f = open(self.config_path, 'w')

        for k in config.keys():
            _type = self.dict_data[k]['type']
            v = config[k]
            match _type:
                case 'continuous':
                    v = round(v, 2)
                    v = str(v)
                case 'numerical':
                    v = str(v)
                case 'binary':
                    _items = self.dict_data[k]['range'].split(',')
                    v = _items[v]
                case 'categorical':
                    _items = self.dict_data[k]['range'].split(',')
                    v = _items[v]
            if log_flag:
                logging.info(f'{k}={v}')
            f.writelines(f'{k}={v}\n')
    
    def random_sampling_configuration(self) -> Configuration:
        return self.cs.sample_configuration()
    
    def apply_and_run_configuration(self, load=bool):
        self.apply_configuration()
        self.run_configuration(load)
