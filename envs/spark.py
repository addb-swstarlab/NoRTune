import os
import torch
import logging
import pandas as pd

import envs.params as p
from statistics import mean

class SparkEnv:
    def __init__(
        self,
        csv_path: str = p.SPARK_CONF_INFO_CSV_PATH,
        config_path: str = p.SPARK_CONF_PATH,
        workload: str = None,
        alter: bool = True
    ):
        self.config_path=config_path
        
        csv_data = pd.read_csv(csv_path, index_col=0)
        ## TODO: nan --> 'blank', modify to process nan values
        # csv_data = pd.read_csv(csv_path, index_col=0, keep_default_na=False)
        self.dict_data = csv_data.to_dict(orient='index')
        
        self.workload = workload if workload is not None else 'join'
        
        if alter:
            self._alter_hibench_configuration()
            # self._get_result_from_default_configuration()
        
    def _alter_hibench_configuration(self):
        self.workload_size = {
            'aggregation': 'huge', #'gigantic', #'huge',
            'join': 'huge', #'huge',
            'scan': 'huge', #'huge',
            'wordcount': 'large',
            'terasort': 'large', 
            'bayes': 'huge', #'huge',
            'kmeans': 'large',
            'pagerank': 'large',
            'svm': 'small',
            'nweight': 'small',
        }
        HIBENCH_CONF_PATH = os.path.join(p.DATA_FOLDER_PATH, f'{self.workload_size[self.workload]}_hibench.conf')
        logging.info("Altering hibench workload scale..")
        logging.info(f"Workload ***{self.workload}*** need ***{self.workload_size[self.workload]}*** size..")
        os.system(f'scp {HIBENCH_CONF_PATH} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}/hibench.conf')


    def apply_configuration(self, config_path=None):
        config_path = self.config_path if config_path is None else config_path
        
        logging.info("Applying created configuration to the remote Spark server.. 💨💨")
        os.system(f'scp {config_path} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}/add-spark.conf')
        
    def run_configuration(self):
        """
            TODO:
            !!A function to Save configuration should be implemented on other files!!
        """
        
        exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_{self.workload}.sh"')
        # exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_bayes.sh"')
        if exit_code > 0:
            logging.warning("💀Failed benchmarking!!")
            logging.warning("UNVALID CONFIGURATION!!")
            self.fail_conf_flag = True
        else:
            logging.info("🎉Successfully finished benchmarking")
            self.fail_conf_flag = False
                        
    def get_results(self) -> float:
        logging.info("Getting result files..")
        if self.fail_conf_flag:
            duration = 10000
            tps = 0.1
        else:
            os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/report_transport.sh"')
            f = open(p.HIBENCH_REPORT_PATH, 'r')
            report = f.readlines()
            f.close()
            
            duration = report[-1].split()[-3]
            tps = report[-1].split()[-2]
        logging.info(f"The recorded results are.. Duration: {duration} s Throughput: {tps} bytes/s")
        # return float(duration), float(tps)
        return float(duration)
    
    # Clear hdfs storages in the remote Spark nodes
    def clear_spark_storage(self):
        exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/clear_hibench.sh"')
        if exit_code > 0:
            logging.warning("💀Failed cleaning Spark Storage!!")
        else:
            logging.info("🎉Successfully cleaning Spark Storage")
    
    # Get the result of default configuration..
    def _get_result_from_default_configuration(self): 
        logging.info("💻Benchmarking the default configuration...")
        # This default configuration occurs an error in benchmarking
        self.apply_configuration(config_path=p.SPARK_DEFAULT_CONF_PATH)
        
        res_ = []
        for _ in range(p.BENCHMARKING_REPETITION):
            self.run_configuration()
            res_.append(self.get_results())
            
        self.def_res = mean(res_)
        logging.info(f"Default duration (s) is {self.def_res}")
    
    def calculate_improvement_from_default(self, best_fx):
        # default_fx = self._get_result_from_default_configuration()
        default_fx = self.def_res
        if isinstance(best_fx, torch.Tensor):
            best_fx = best_fx.item()
            
        improve_ratio = round((default_fx - best_fx)/default_fx * 100, 2)
        logging.info("=============================================================")
        logging.info(f"🎯 Improvement rate from default results.. {improve_ratio}%")
        logging.info(f"Default result is {default_fx} and Best result is {best_fx}")
        logging.info("=============================================================")