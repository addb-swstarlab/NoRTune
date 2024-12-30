import os, json, logging
from statistics import mean, stdev
from ConfigSpace import Configuration
from smac.scenario import Scenario
from smac import HyperparameterOptimizationFacade
from smac import BlackBoxFacade
from smac.model.random_forest.random_forest import RandomForest
from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.random_design.probability_design import ProbabilityRandomDesign

from envs.params import BOUNCE_PARAM as bp
import envs.params as p
from envs.utils import get_foldername
from others.benchmarks import Benchmark
from others.adapters.acquisition_function import AEI

class Baselines:
    def __init__(
        self, 
        embedding_method: str, # ['rembo', 'hesbo', 'ddpg']
        optimizer_method: str, # ['smac', 'bo']
        benchmark: Benchmark,
        maximum_number_evaluations: int = bp["maximum_number_evaluations"],
        rand_percentage: float = 0.1,
        n_estimators: int = 100,
        acquisition_function: str = 'ei',
        is_tps: bool = False,
    ):
        self.embedding_method = embedding_method
        self.optimizer_method = optimizer_method
        self.benchmark = benchmark
        self.rand_percentage = rand_percentage
        self.n_estimators = n_estimators
        self.input_space = self.benchmark.input_space
        self.acquisition_function = acquisition_function
        self.is_tps = is_tps
        
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
        self.results_dir = get_foldername(os.path.join(path, f"{self.embedding_method}_results"))
        os.makedirs(self.results_dir, exist_ok=True)
        logging.info(f"Results are saved in .. {self.results_dir}")
        f = open(os.path.join(self.results_dir, 'workload.txt'), 'w')
        f.writelines(f"{self.benchmark.workload} {self.benchmark.workload_size}")
        f.close()    
    
    # def _get_input_space(self):
    #     if self.method == 'ddpg':
    #         return self.benchmark.input_space
    #     elif self.method in ['rembo', 'hesbo']:
    #         return self.benchmark.input_space

    def _init_observations(self):
        self._x = []
        self._y = []
        self._repeated_y = []
    
    def add_observation(self, config: Configuration, res: float, r_res: list=None):
        if self.embedding_method in ['rembo', 'hesbo']:
            x: dict = self.benchmark.embedding_adapter.unproject_point(config)
        else:
            x = config.get_dictionary()
        
        self._x.append(x)
        self._y.append(res)
        if r_res is not None:
            self._repeated_y.append(r_res)
        
    
    def get_tuner(self):
        if self.optimizer_method == 'smac':
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
        elif self.optimizer_method == 'bo':
            optimizer = BlackBoxFacade(
                scenario=self.scenario,
                target_function=self.target_function,
                model=GaussianProcess(
                    configspace=self.input_space,
                    kernel=BlackBoxFacade.get_kernel(scenario=self.scenario),
                    normalize_y=True
                ),
                random_design=ProbabilityRandomDesign(self.rand_percentage),
                acquisition_function=AEI(xi=0.0) if self.acquisition_function == 'aei' else None
            )
        else:
            assert False, "Please define optimizer method to smac or bo."
        return optimizer


    def target_function(self, x: Configuration, seed: int=0) -> float:
        r_fx = self.benchmark.evaluate(x, seed=seed, load=True, repeat=self.repeat)
        if self.is_tps:
            r_fx = -r_fx
        mean_fx = mean(r_fx)

        self.add_observation(x, mean_fx, r_fx)
        logging.info(f"🚀 Iteration {self.cnt}: the evaluated results are {r_fx} / Mean = {mean_fx:.3f}")
        
        if mean_fx < self.best_res:
            logging.info(f"🔔 Best result is updated!! : {self.best_res:.3f} --> {mean_fx:.3f}")
            self.best_res = mean_fx
            
            f = open(p.CONF_PATH, 'r')
            self.best_config = f.readlines()
            self.best_x = x
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
        
        logging.info(f"✨✨✨ Evaluating best x... # of repetitions = {p.BENCHMARKING_REPETITION} ✨✨✨")
        
        best_ys = self.benchmark.evaluate(self.best_x, load=True, repeat=p.BENCHMARKING_REPETITION)
        
        # best_ys = []
        # for _ in range(p.BENCHMARKING_REPETITION):
        #     best_y = self.benchmark.evaluate(self.best_x, load=True if _ == 0 else False)
        #     best_ys.append(best_y)
        logging.info(f"Results = {best_ys} , Mean = {mean(best_ys):.3f} (±{stdev(best_ys):.3f})")
        
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
