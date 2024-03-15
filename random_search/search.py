import os
import json
import logging
from statistics import mean
# from random_search.benchmarks import SparkBench
from envs.utils import get_foldername
from envs.params import BOUNCE_PARAM as bp, BENCHMARKING_REPETITION

class RandomSearch: # Random Optimizer?
    def __init__(
        self,
        benchmark,
        maximum_number_evaluations: int = bp["maximum_number_evaluations"] # 100
    ):
        # self.benchmark = benchmark if benchmark is not None else SparkBench()
        self.benchmark = benchmark
        self.maximum_number_evaluations = maximum_number_evaluations                        
        self._set_result_dir()
    
    def _set_result_dir(self, path='/home/jieun/SparkTuning/random_search/results'):
        self.results_dir = get_foldername(path)
        os.makedirs(self.results_dir, exist_ok=True)
        logging.info(f"Results are saved in .. {self.results_dir}")
    
    def run(self):
        logging.info("Start Random Search!!")       
        best_config = None
        best_res = 10000
        
        configs = []
        results = []
        
        for i in range(self.maximum_number_evaluations):
            sampled_config = self.benchmark.random_sampling_configuration()
            self.benchmark.save_configuration_file(sampled_config)
            
            res_ = []
            for _ in range(BENCHMARKING_REPETITION):
                self.benchmark.apply_and_run_configuration()
                res_.append(self.benchmark.get_results()) 
            res = mean(res_)
           
            logging.info(f"[{i}/{self.maximum_number_evaluations}]!!!!!!!!!!!!!!Results:{res:.3f}!!!!!!!!!!!!!!")
            
            if res < best_res:
                logging.info(f"🎉 Best result is updated!! : {best_res:.3f} --> {res:.3f}")
                best_res = res
                
                # best_config = sampled_config
                f = open('/home/jieun/SparkTuning/data/add-spark.conf', 'r')
                best_config = f.readlines()    
                
            configs.append(sampled_config.get_dictionary())
            results.append(res)
        
        # Save history.. configs and results
        with open(os.path.join(self.results_dir, 'configs.json'), 'w') as f:
            json.dump(configs, f)
        
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f)
                    
        logging.info("............................")
        logging.info("........Best results........")
        logging.info(f"{best_res} s")
        logging.info(".....Best Configuration.....")
        for l in best_config:
            logging.info(l)
        logging.info("......................")