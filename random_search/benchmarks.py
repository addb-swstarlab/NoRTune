import os
import logging
import pandas as pd

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter

import envs.params as p
from envs.spark import SparkEnv

class SparkBench(SparkEnv):
    def __init__(
        self,
        workload: str = None,
        alter: bool = True
    ):
        super().__init__(workload=workload, alter=alter)
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
    
    def save_configuration_file(self, config: Configuration, log_flag=False):
        logging.info(f"Save configuration to {self.config_path} 💨")
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
    
    # def apply_configuration(self):
    #     logging.info("Applying created configuration to the remote Spark server..")
    #     os.system(f'scp {self.config_path} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}')
    #     exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_join.sh"')
    #     # exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_bayes.sh"')
    #     if exit_code > 0:
    #         logging.error("Failed benchmarking!!")
    #         logging.error("UNVALID CONFIGURATION!!")
    #         self.fail_conf_flag = True
    #     else:
    #         logging.info("Successflly finished benchmarking")
    #         self.fail_conf_flag = False
    
    # def get_results(self):
    #     logging.info("Getting result files..")
    #     if self.fail_conf_flag:
    #         duration = 10000
    #         tps = 0.1
    #     else:
    #         os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/report_transport.sh"')
    #         f = open(p.HIBENCH_REPORT_PATH, 'r')
    #         report = f.readlines()
    #         f.close()
            
    #         duration = report[-1].split()[-3]
    #         tps = report[-1].split()[-2]
        
    #     # return float(duration), float(tps)
    #     return float(duration)