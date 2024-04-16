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
        workload_size: str = None,
        alter: bool = True,
        debugging: bool = False
    ):
        self.config_path=config_path
        
        csv_data = pd.read_csv(csv_path, index_col=0)
        ## TODO: nan --> 'blank', modify to process nan values
        # csv_data = pd.read_csv(csv_path, index_col=0, keep_default_na=False)
        self.dict_data = csv_data.to_dict(orient='index')
        
        self.workload = workload if workload is not None else 'join'
        
        self.debugging = debugging
        
        self.alter = False if self.debugging else alter
        
        self.workloads_size = {
            'aggregation': 'large', #'huge',
            'join': 'huge',
            'scan': 'huge',
            'wordcount': 'large',
            'terasort': 'large',  
            'bayes': 'huge',
            'kmeans': 'large',
            'pagerank': 'large',
            'svm': 'large',
            'nweight': 'small',
        }        
        
        self.start_dataproc()
        
        if self.alter:
            self._alter_hibench_configuration(workload_size)
            # self._get_result_from_default_configuration()
        self.fail_conf_flag = False
        
    def _alter_hibench_configuration(self, workload_size=None):
        if workload_size is None:
            workload_size = self.workloads_size[self.workload]
            
        HIBENCH_CONF_PATH = os.path.join(p.DATA_FOLDER_PATH, f'{workload_size}_hibench.conf')
        logging.info("Altering hibench workload scale..")
        logging.info(f"Workload ***{self.workload}*** with ***{workload_size}*** size..")
        os.system(f'scp {HIBENCH_CONF_PATH} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}/hibench.conf')


    def apply_configuration(self, config_path=None):
        if self.debugging:
            logging.info("DEBUGGING MODE, skipping to apply the given configuration")
        else:
            self._apply_configuration(config_path)
            
    def run_configuration(self):
        if self.debugging:
            logging.info("DEBUGGING MODE, skipping to benchmark the given configuration")
        else:
            self._run_configuration()
            
    def get_results(self):
        if self.debugging:
            logging.info("DEBUGGING MODE, getting results from the local report file..")
            
            f = open(p.HIBENCH_REPORT_PATH, 'r')
            report = f.readlines()
            f.close()
            
            from random import sample
            rand_idx = sample(range(1, len(report)),1)[0]
            
            duration = report[rand_idx].split()[-3]
            tps = report[rand_idx].split()[-2]
            logging.info(f"DEBUGGING MODE, the recorded results are.. Duration: {duration} s Throughput: {tps} bytes/s")
            # return float(duration), float(tps)
            return float(duration)
        else:
            return self._get_results()
    
    def _apply_configuration(self, config_path=None):
        config_path = self.config_path if config_path is None else config_path
        
        logging.info("Applying created configuration to the remote Spark server.. 💨💨")
        os.system(f'scp {config_path} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}/add-spark.conf')
        
    def _run_configuration(self):
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
                        
    def _get_results(self) -> float:
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
        if self.debugging:
            logging.info("[Google Cloud Platform|Dataproc] 🛑 Skipping cleaning Spark storage!!")
        else:
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
        
    def start_dataproc(self):
        if self.debugging:
            logging.info("[Google Cloud Platform|Dataproc] 🛑 Skipping start Spark instances")
        else:
            logging.info("[Google Cloud Platform|Dataproc] 🔥 Start Spark instances")
            os.system(p.GCP_DATAPROC_START_COMMAND)
        
    def stop_dataproc(self):
        if self.debugging:
            logging.info("[Google Cloud Platform|Dataproc] 🛑 Skipping stop Spark instances")
        else:
            logging.info("[Google Cloud Platform|Dataproc] ⛔ Stop Spark instances")
            os.system(p.GCP_DATAPROC_STOP_COMMAND)