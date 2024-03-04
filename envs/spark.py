import os
import logging
import pandas as pd

import envs.params as p

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
        
    def _alter_hibench_configuration(self):
        workload_size = {
            'aggregation': 'huge', #'gigantic', #'huge',
            'join': 'huge', #'huge',
            'scan': 'huge', #'huge',
            'wordcount': 'large',
            'terasort': 'large',
            'bayes': 'huge', #'huge',
            'kmeans': 'large',
            'pagerank': 'large'
        }
        HIBENCH_CONF_PATH = os.path.join(p.DATA_FOLDER_PATH, f'{workload_size[self.workload]}_hibench.conf')
        logging.info("Altering hibench workload scale..")
        logging.info(f"Workload ***{self.workload}*** need ***{workload_size[self.workload]}*** size..")
        os.system(f'scp {HIBENCH_CONF_PATH} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}/hibench.conf')
        
    def apply_configuration(self):
        """
            TODO:
            !!A function to Save configuration should be implemented on other files!!
        """
        logging.info("Applying created configuration to the remote Spark server.. 💨💨")
        os.system(f'scp {self.config_path} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}')

        exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_{self.workload}.sh"')
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