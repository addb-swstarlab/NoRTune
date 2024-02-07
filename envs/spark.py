import os
import logging
import pandas as pd

import envs.params as p

class SparkEnv:
    def __init__(
        self,
        csv_path: str = p.SPARK_CONF_INFO_CSV_PATH,
        config_path: str = p.SPARK_CONF_PATH,
        benchmark: str = None
    ):
        self.config_path=config_path
        
        csv_data = pd.read_csv(csv_path, index_col=0)
        self.dict_data = csv_data.to_dict(orient='index')
        
        self.benchmark = benchmark if benchmark is not None else 'join'
        
        self._alter_hibench_configuration()
        
    def _alter_hibench_configuration(self):
        benchmark_size = {
            'aggregation': 'huge',
            'join': 'gigantic', #'huge',
            'scan': 'huge',
            'wordcount': 'large',
            'terasort': 'large',
            'bayes': 'huge',
            'kmeans': 'large',
            'pagerank': 'large'
        }
        HIBENCH_CONF_PATH = os.path.join(p.DATA_FOLDER_PATH, f'{benchmark_size[self.benchmark]}_hibench.conf')
        logging.info("Altering hibench workload scale..")
        logging.info(f"Workload {self.benchmark} need {benchmark_size[self.benchmark]} size..")
        os.system(f'scp {HIBENCH_CONF_PATH} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}/hibench.conf')
        
        
    def apply_configuration(self):
        """
            TODO:
            !!A function to Save configuration should be implemented on other files!!
        """
        logging.info("Applying created configuration to the remote Spark server..")
        os.system(f'scp {self.config_path} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}')
        exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_{self.benchmark}.sh"')
        # exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_bayes.sh"')
        if exit_code > 0:
            logging.warning("Failed benchmarking!!")
            logging.warning("UNVALID CONFIGURATION!!")
            self.fail_conf_flag = True
        else:
            logging.info("Successflly finished benchmarking")
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
        
        # return float(duration), float(tps)
        return float(duration)