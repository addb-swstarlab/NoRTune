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
        # workload_size: str = None,
        # alter: bool = True,
        debugging: bool = False
    ):
        self.config_path=config_path
        
        csv_data = pd.read_csv(csv_path, index_col=0)
        ## TODO: nan --> 'blank', modify to process nan values
        # csv_data = pd.read_csv(csv_path, index_col=0, keep_default_na=False)
        self.dict_data = csv_data.to_dict(orient='index')
        
        self.workload = workload if workload is not None else 'ycsb-a'
        
        self.debugging = debugging
        
        # self.alter = False if self.debugging else alter
        
        self.workload_size = 'none'
        
        self.timeout = 400
        # self.timeout = 1000 if workload_size in ["tiny", "small", "large"] else 2000
        # self.workloads_size = {
        #     'aggregation': 'large', #'huge',
        #     'join': 'huge',
        #     'scan': 'huge',
        #     'wordcount': 'large',
        #     'terasort': 'large',  
        #     'bayes': 'huge',
        #     'kmeans': 'large',
        #     'pagerank': 'large',
        #     'svm': 'large',
        #     'nweight': 'small',
        # }        
        
        # self.start_dataproc()
        
        # if self.alter:
        #     self._alter_hibench_configuration(self.workload_size)
        #     # self._get_result_from_default_configuration()
        self.fail_conf_flag = False
        
        self.result_logs = None
        
    # def _alter_hibench_configuration(self, workload_size=None):
    #     if workload_size is None:
    #         workload_size = 'large' # self.workloads_size[self.workload]
            
    #     HIBENCH_CONF_PATH = os.path.join(p.DATA_FOLDER_PATH, f'{workload_size}_hibench.conf')
    #     logging.info("Altering hibench workload scale..")
    #     logging.info(f"Workload ***{self.workload}*** with ***{workload_size}*** size..")
    #     os.system(f'scp {HIBENCH_CONF_PATH} {p.MASTER_ADDRESS}:{p.MASTER_CONF_PATH}/hibench.conf')


    def apply_configuration(self, config_path=None):
        if self.debugging:
            logging.info("DEBUGGING MODE, skipping to apply the given configuration")
        else:
            self._apply_configuration(config_path)
            
    def run_configuration(self, load:bool):
        if self.debugging:
            logging.info(f"DEBUGGING MODE, skipping to benchmark the given configuration")
            # start = time.time()
            # logging.info(f"DEBUGGING MODE, skipping to benchmark the given configuration --> ### LOAD? {load}")
            # end = time.time()
            # logging.info(f"DEBUGGING MODE, [HiBench]⏱ data loading takes {end - start} s ⏱")
        else:
            self._run_configuration(load)
            
    def get_results(self):
        if self.debugging:
            # logging.info("DEBUGGING MODE, getting results from the local report file..")
            
            # f = open(p.HIBENCH_REPORT_PATH, 'r')
            # report = f.readlines()
            # f.close()
            
            # from random import sample
            # rand_idx = sample(range(1, len(report)),1)[0]
            
            # duration = report[rand_idx].split()[-3]
            # tps = report[rand_idx].split()[-2]
            # logging.info(f"DEBUGGING MODE, the recorded results are.. Duration: {duration} s Throughput: {tps} bytes/s")
            # # return float(duration), float(tps)
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
            print(run_command)
            assert False
        elif self.workload == 'ycsb-b':
            run_command = f'timeout {self.timeout} sshpass -p {p.POSTGRES_SERVER_PASSWSD} ssh {p.POSTGRES_SERVER_ADDRESS} {p.POSTGRES_SERVER_POSTGRES_PATH}/run_workloadb.sh'
        
        logging.info("Running benchmark..")
        result = subprocess.run(run_command, shell=True, capture_output=True, text=True)
        logging.info("🎉Successfully finished benchmarking")
        
        self.result_logs = result.stdout
        self.result_errs = result.stderr
        # # if load:
        # #     start = time.time()
        # #     os.system(f'timeout {self.timeout} ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/prepare_wk/load_{self.workload}.sh"')
        # #     # os.system(f'timeout 1000 ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/prepare_wk/load_{self.workload}.sh"')
        # #     end = time.time()
        # #     logging.info(f"[HiBench] data loading (seconds) takes {end - start}")
        
        # # exit_code = os.system(f'timeout 1000 ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_wk/run_{self.workload}.sh"')
        # exit_code = os.system(f'timeout {self.timeout} ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_wk/run_{self.workload}.sh"')
        # # exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_{self.workload}.sh"')
        # # exit_code = os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/run_bayes.sh"')
        if len(self.result_errs) > 0:
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
        
        # logging.info("Getting result files..")
        # if self.fail_conf_flag:
        #     duration = 10000
        #     tps = 0.1
        # else:
        #     os.system(f'ssh {p.MASTER_ADDRESS} "bash --noprofile --norc -c scripts/report_transport.sh"')
        #     f = open(p.HIBENCH_REPORT_PATH, 'r')
        #     report = f.readlines()
        #     f.close()
            
        #     duration = report[-1].split()[-3]
        #     tps = report[-1].split()[-2]
        # logging.info(f"The recorded results are.. Duration: {duration} s Throughput: {tps} bytes/s")
        # return float(duration), float(tps)
        return float(tps)
    
    def _restart_postgres(self):
        logging.info("Restart PostgreSQL service to apply configuration..")
        # os.system(f"sshpass -p {p.POSTGRES_SERVER_PASSWSD} ssh {p.POSTGRES_SERVER_ADDRESS} 'echo {p.POSTGRES_SERVER_PASSWSD} | sudo -S systemctl restart postgresql'")
        os.system(f"sshpass -p {p.POSTGRES_SERVER_PASSWSD} ssh {p.POSTGRES_SERVER_ADDRESS} 'echo {p.POSTGRES_SERVER_PASSWSD} | sudo -S {p.POSTGRES_SERVER_POSTGRES_PATH}/reset_benchmark.sh'")
        logging.info("Restart PostgreSQL service finished..")