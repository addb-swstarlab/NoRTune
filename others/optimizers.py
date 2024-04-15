import os, json, logging
from statistics import mean
from ConfigSpace import Configuration
from smac.scenario import Scenario
from smac import HyperparameterOptimizationFacade
from smac import BlackBoxFacade
from smac.model.random_forest.random_forest import RandomForest
from smac.random_design.probability_design import ProbabilityRandomDesign

from envs.params import BOUNCE_PARAM as bp
import envs.params as p
from envs.utils import get_foldername
from others.benchmarks import Benchmark

class Baselines:
    def __init__(
        self, 
        method: str, # ['rembo', 'hesbo', 'ddpg']
        benchmark: Benchmark,
        maximum_number_evaluations: int = bp["maximum_number_evaluations"],
        rand_percentage: float = 0.1,
        n_estimators: int = 100,
    ):
        self.method = method
        self.benchmark = benchmark
        self.rand_percentage = rand_percentage
        self.n_estimators = n_estimators
        self.input_space = self._get_input_space()
        
        self._init_observations()
        self._set_result_dir()
        
        self.scenario = Scenario(
            configspace=self.input_space,
            objectives="quality",
            n_trials=maximum_number_evaluations,
            deterministic=True,
            output_directory=self.results_dir,
        )
    
    def _set_result_dir(self, path=os.path.join(p.HOME_PATH, p.PROJECT_NAME)):
        self.results_dir = get_foldername(os.path.join(path, f"{self.method}_results"))
        os.makedirs(self.results_dir, exist_ok=True)
        logging.info(f"Results are saved in .. {self.results_dir}")
        f = open(os.path.join(self.results_dir, 'workload.txt'), 'w')
        f.writelines(f"{self.benchmark.workload} {self.benchmark.workloads_size[self.benchmark.workload]}")
        f.close()    
    
    def _get_input_space(self):
        if self.method == 'ddpg':
            return self.benchmark.input_space
        elif self.method in ['rembo', 'hesbo']:
            return self.benchmark.input_space

    def _init_observations(self):
        self._x = []
        self._y = []
        self._repeated_y = []
    
    def add_observation(self, config: Configuration, res: float, r_res: list=None):
        if self.method in ['rembo', 'hesbo']:
            x: dict = self.benchmark.embedding_adapter.unproject_point(config)
        else:
            x = config.get_dictionary()
        
        self._x.append(x)
        self._y.append(res)
        if r_res is not None:
            self._repeated_y.append(r_res)
        
    
    def get_tuner(self):
        optimizer = HyperparameterOptimizationFacade(
            scenario=self.scenario,
            target_function=self.target_function,
            model=RandomForest(
                    configspace=self.input_space,
                    log_y=False,
                    n_trees=self.n_estimators,
                    ratio_features=1,
                    min_samples_split=2,
                    min_samples_leaf=3,
                ),
            random_design=ProbabilityRandomDesign(self.rand_percentage),
        )
        return optimizer

    def target_function(self, x: Configuration, seed: int=0) -> float:
        r_fx = []
        for _ in range(p.BENCHMARKING_REPETITION):
            fx = self.benchmark.evaluate(x, seed=seed)
            r_fx.append(fx)
        mean_fx = mean(r_fx)

        self.add_observation(x, mean_fx, r_fx)
        logging.info(f"🚀 Iteration {self.cnt}: the evaluated results are {r_fx} / Mean = {mean_fx:.3f}")
        
        if mean_fx < self.best_res:
            logging.info(f"🔔 Best result is updated!! : {self.best_res:.3f} --> {mean_fx:.3f}")
            self.best_res = mean_fx
            
            f = open(p.SPARK_CONF_PATH, 'r')
            self.best_config = f.readlines()
        else:    
            logging.info(f"💦 Best function value is still {self.best_res}")
        
        self.cnt += 1
        
        return mean_fx
    
    def run(self):
        self.cnt = 0
        self.best_config = None
        self.best_res = 10000
        
        self.optimizer = self.get_tuner()
        
        inc = self.optimizer.optimize()
        
        logging.info("............................")
        logging.info("........Best results........")
        logging.info(f"{self.best_res} s")
        logging.info(".....Best Configuration.....\n")
        logging.info(''.join(self.best_config))
        # for l in best_config:
        #     logging.info(l)
        logging.info("......................")
        
        self._save_observations_to_json()
        
    def _save_observations_to_json(self):
        logging.info("💬 saving observations to json files..")
        with open(os.path.join(self.results_dir, 'configs.json'), 'w') as f:
            json.dump(self._x, f)
        
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(self._y, f)
            
        if self._repeated_y is not None:
            with open(os.path.join(self.results_dir, 'repeated_results.json'), 'w') as f:
                json.dump(self._repeated_y, f)
                    
        # with open(os.path.join(self.results_dir, 'repeated_configs.json'), 'w') as f:
        #     json.dump(repeated_configs, f)
        
        # with open(os.path.join(self.results_dir, 'repeated_results.json'), 'w') as f:
        #     json.dump(repeated_results, f)