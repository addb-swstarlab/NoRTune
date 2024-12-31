import os, time
import subprocess
import logging
import pandas as pd

import envs.params as p
from statistics import mean

class PostgresEnv:
    def __init__(
        self,
        csv_path: str = p.POSTGRES_CONF_INFO_CSV_PATH,
        config_path: str = p.CONF_PATH,
        workload: str = None,
        debugging: bool = False
    ):
        self.config_path=config_path
        
        csv_data = pd.read_csv(csv_path, index_col=0)
        ## TODO: nan --> 'blank', modify to process nan values
        # csv_data = pd.read_csv(csv_path, index_col=0, keep_default_na=False)
        self.dict_data = csv_data.to_dict(orient='index')
        
        self.workload = workload if workload is not None else 'ycsb-a'
        
        self.debugging = debugging
               
        self.workload_size = 'none'
        
        self.timeout = 400

        self.fail_conf_flag = False
        
        self.result_logs = None

    def apply_configuration(self, config_path=None):
        if self.debugging:
            logging.info("DEBUGGING MODE, skipping to apply the given configuration")
        else:
            self._apply_configuration(config_path)
            
    def run_configuration(self, load:bool):
        if self.debugging:
            logging.info(f"DEBUGGING MODE, skipping to benchmark the given configuration")
        else:
            self._run_configuration(load)
            
    def get_results(self):
        if self.debugging:
            from random import random
            tps = random() * 1000
            return float(tps)
        else:
            return self._get_results()
    
    def _apply_configuration(self, config_path=None):
        config_path = self.config_path if config_path is None else config_path
        
        logging.info("Applying created configuration to the remote PostgreSQL server.. 💨💨")
        os.system(f'sshpass -p {p.POSTGRES_SERVER_PASSWSD} scp {config_path} {p.POSTGRES_SERVER_ADDRESS}:{p.POSTGRES_SERVER_CONF_PATH}/add-postgres.conf')
        self._restart_postgres()
        
    def _run_configuration(self, load:bool):       
        if self.workload == 'ycsb-a':
            run_command = f'timeout {self.timeout} sshpass -p {p.POSTGRES_SERVER_PASSWSD} ssh {p.POSTGRES_SERVER_ADDRESS} {p.POSTGRES_SERVER_POSTGRES_PATH}/run_workloada.sh'
        elif self.workload == 'ycsb-b':
            run_command = f'timeout {self.timeout} sshpass -p {p.POSTGRES_SERVER_PASSWSD} ssh {p.POSTGRES_SERVER_ADDRESS} {p.POSTGRES_SERVER_POSTGRES_PATH}/run_workloadb.sh'
        
        logging.info("Running benchmark..")
        result = subprocess.run(run_command, shell=True, capture_output=True, text=True)
        
        self.result_logs = result.stdout
        self.result_exit_code = result.returncode

        if self.result_exit_code > 0:
            logging.warning("💀Failed benchmarking!!")
            logging.warning("UNVALID CONFIGURATION!!")
            self.fail_conf_flag = True
        else:
            logging.info("🎉Successfully finished benchmarking")
            self.fail_conf_flag = False
                        
    def _get_results(self) -> float:
        if self.fail_conf_flag:
            duration = 0
            tps = 0.1
            logging.info(f"[💀 ERROR OCCURRED 💀]The recorded results are.. Duration: {duration} s Throughput: {tps} bytes/s")
        else:
            duration, tps = [s.split()[-1] for s in self.result_logs.split('\n') if '[OVERALL]' in s]
            logging.info(f"The recorded results are.. Duration: {duration} s Throughput: {tps} bytes/s")

        return float(tps)
    
    def _restart_postgres(self):
        logging.info("Restart PostgreSQL service to apply configuration..")
        # os.system(f"sshpass -p {p.POSTGRES_SERVER_PASSWSD} ssh {p.POSTGRES_SERVER_ADDRESS} 'echo {p.POSTGRES_SERVER_PASSWSD} | sudo -S systemctl restart postgresql'")
        # os.system(f"sshpass -p {p.POSTGRES_SERVER_PASSWSD} ssh {p.POSTGRES_SERVER_ADDRESS} 'echo {p.POSTGRES_SERVER_PASSWSD} | sudo -S {p.POSTGRES_SERVER_POSTGRES_PATH}/reset_benchmark.sh'")
        os.system(f"sshpass -p {p.POSTGRES_SERVER_PASSWSD} ssh {p.POSTGRES_SERVER_ADDRESS} '{p.POSTGRES_SERVER_POSTGRES_PATH}/reset_benchmark.sh'")
        logging.info("Restart PostgreSQL service finished..")