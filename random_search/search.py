import os
import json
import logging
import numpy as np
from statistics import mean, stdev
from typing import Union
# from random_search.benchmarks import SparkBench
from envs.utils import get_foldername
from envs.params import BOUNCE_PARAM as bp, BENCHMARKING_REPETITION, CONF_PATH, HOME_PATH, PROJECT_NAME
from random_search.benchmarks import SparkBench, PostgresBench

class RandomSearch: # Random Optimizer?
    def __init__(
        self,
        benchmark : Union[SparkBench, PostgresBench],
        maximum_number_evaluations: int = bp["maximum_number_evaluations"], # 100,
        is_tps : bool = False
    ):
        # self.benchmark = benchmark if benchmark is not None else SparkBench()
        self.benchmark = benchmark
        self.maximum_number_evaluations = maximum_number_evaluations                        
        self._set_result_dir()
        self.is_tps = is_tps
    
    def _set_result_dir(self, path=os.path.join(HOME_PATH, PROJECT_NAME, 'random_search/results')):
        self.results_dir = get_foldername(path)
        os.makedirs(self.results_dir, exist_ok=True)
        logging.info(f"Results are saved in .. {self.results_dir}")
        f = open(os.path.join(self.results_dir, 'workload.txt'), 'w')
        f.writelines(f"{self.benchmark.workload} {self.benchmark.workload_size}")
        f.close()
            
    def run(self):
        logging.info("Start Random Search!!")       
        best_config = None
        best_res = 10000
        
        configs = []
        results = []
        
        repeated_configs = []
        repeated_results = []
        
        for i in range(self.maximum_number_evaluations):
            sampled_config = self.benchmark.random_sampling_configuration()
            self.benchmark.save_configuration_file(sampled_config)
            sampled_config = dict(sampled_config)
            sampled_config = {k: int(v) if isinstance(v, np.integer) else v for k, v in sampled_config.items()}
            
            res_ = []
            self.benchmark.apply_configuration()
            load = True
            for _ in range(BENCHMARKING_REPETITION):
                self.benchmark.run_configuration(load)
                load = False
                # self.benchmark.apply_and_run_configuration(load=True if _ == 0 else False)
                if self.is_tps:
                    res_.append(-self.benchmark.get_results()) # Higher is better, tps
                else:
                    res_.append(self.benchmark.get_results()) # Lower is better, latency or runtime
                # repeated_configs.append(dict(sampled_config))
                repeated_configs.append(sampled_config)
                repeated_results.append(res_)
            res = mean(res_)
           
            logging.info(f"[{i}/{self.maximum_number_evaluations}] Results = {res_}, MEAN = {res:.3f}")
            
            if res < best_res:
                logging.info(f"🎉 Best result is updated!! : {best_res:.3f} --> {res:.3f}")
                best_res = res
                
                # best_config = sampled_config
                f = open(CONF_PATH, 'r')
                best_config = f.readlines() 
                best_config_dict = sampled_config 
            
            # configs.append(sampled_config.get_dictionary())
            configs.append(sampled_config)
            results.append(res)
        
        # Save history.. configs and results
        with open(os.path.join(self.results_dir, 'configs.json'), 'w') as f:
            json.dump(configs, f)
        
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f)
                    
        with open(os.path.join(self.results_dir, 'repeated_configs.json'), 'w') as f:
            json.dump(repeated_configs, f)
        
        with open(os.path.join(self.results_dir, 'repeated_results.json'), 'w') as f:
            json.dump(repeated_results, f)
                    
                    
        logging.info("............................")
        logging.info("........Best results........")
        logging.info(f"{best_res} s")
        logging.info(".....Best Configuration.....")
        logging.info(''.join(best_config))
        # for l in best_config:
        #     logging.info(l)
        logging.info("......................")
        
        logging.info(f"✨✨✨ Evaluating best x... # of repetitions = {BENCHMARKING_REPETITION} ✨✨✨")
        best_ys = []
        self.benchmark.save_configuration_file(best_config_dict)
        self.benchmark.apply_configuration()
        load = True
        for _ in range(BENCHMARKING_REPETITION):
            self.benchmark.run_configuration(load)
            load = False
            # self.benchmark.apply_and_run_configuration(load=True if _ == 0 else False)
            best_ys.append(self.benchmark.get_results()) 
        logging.info(f"Results = {best_ys} , Mean = {mean(best_ys):.3f} (±{stdev(best_ys):.3f})")         
        # self.benchmark.calculate_improvement_from_default(best_res)    
    
    # def run(self):
    #     logging.info("Start Random Search!!")       
    #     best_config = None
    #     best_res = 10000
        
    #     configs = []
    #     results = []
        
    #     for i in range(self.maximum_number_evaluations):
    #         sampled_config = self.benchmark.random_sampling_configuration()
    #         self.benchmark.save_configuration_file(sampled_config)
            
    #         res_ = []
    #         for _ in range(BENCHMARKING_REPETITION):
    #             self.benchmark.apply_and_run_configuration()
    #             res_.append(self.benchmark.get_results()) 
    #         res = mean(res_)
           
    #         logging.info(f"[{i}/{self.maximum_number_evaluations}]!!!!!!!!!!!!!!Results:{res:.3f}!!!!!!!!!!!!!!")
            
    #         if res < best_res:
    #             logging.info(f"🎉 Best result is updated!! : {best_res:.3f} --> {res:.3f}")
    #             best_res = res
                
    #             # best_config = sampled_config
    #             f = open('/home/jieun/SparkTuning/data/add-spark.conf', 'r')
    #             best_config = f.readlines()    
                
    #         configs.append(sampled_config.get_dictionary())
    #         results.append(res)
        
    #     # Save history.. configs and results
    #     with open(os.path.join(self.results_dir, 'configs.json'), 'w') as f:
    #         json.dump(configs, f)
        
    #     with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
    #         json.dump(results, f)
                    
    #     logging.info("............................")
    #     logging.info("........Best results........")
    #     logging.info(f"{best_res} s")
    #     logging.info(".....Best Configuration.....")
    #     logging.info(''.join(best_config))
    #     # for l in best_config:
    #     #     logging.info(l)
    #     logging.info("......................")
        
    #     self.benchmark.calculate_improvement_from_default(best_res)