from bounce.benchmarks import Benchmark
from bounce.util.benchmark import Parameter, ParameterType
from envs.params import BENCHMARKING_REPETITION
from envs.spark import SparkEnv
import torch
import logging
from statistics import mean

class SparkTuning(Benchmark):
    def __init__(self, env: SparkEnv):
        self.env = env
        self.n_features = len(self.env.dict_data)
        logging.info(f"n_features: {self.n_features}")
        # self.fail_conf_flag = False        

        self.config_path = self.env.config_path
        self.dict_data = self.env.dict_data
        
        parameters = self._define_parameters()
        
        super().__init__(parameters=parameters, noise_std=None)

        self.flip = False

            
    def _define_parameters(self) -> list:
        self.discrete_parameters = [] # for boolean parameters
        self.continuous_parameters = [] # for float parameters
        self.numerical_parameters =  [] # for int parameters
        self.categorical_parameters = [] # for categorical parameters
        
        for k in self.dict_data.keys():
            p_type = getattr(ParameterType, self.dict_data[k]['type'].upper())
            p = Parameter(
                name=k,
                type=p_type,
                lower_bound=self.dict_data[k]['min'],
                upper_bound=self.dict_data[k]['max'],
                unit=None if self.dict_data[k]['unit'] != self.dict_data[k]['unit'] else self.dict_data[k]['unit'],
                items=None if self.dict_data[k]['range'] != self.dict_data[k]['range'] else self.dict_data[k]['range'].split(',')
            )
            match p_type:
                case ParameterType.CONTINUOUS:
                    self.continuous_parameters.append(p)
                case ParameterType.NUMERICAL:
                    self.numerical_parameters.append(p)
                case ParameterType.BINARY:
                    self.discrete_parameters.append(p)
                case ParameterType.CATEGORICAL:
                    self.categorical_parameters.append(p)
                    
        parameters = self.discrete_parameters + self.continuous_parameters + self.numerical_parameters + self.categorical_parameters
        return parameters
    
    def save_configuration_file(self, x: torch.Tensor):        
        logging.info(f"Save configuration to {self.config_path} 💨")
        f = open(self.config_path, 'w')
        
        """
        Converting x without categorical variables into the Spark configuration format.
        Categorical variables must be treat different.
        """
        x_wo_cat = x[:self.categorical_indices[0]]
        for i in range(len(x_wo_cat)):
            p = self.parameters[i].name
            v = x_wo_cat[i]
            match self.parameters[i].type:
                case ParameterType.BINARY:
                    v = int(torch.round(v))
                    v = self.parameters[i].items[v]
                case ParameterType.CONTINUOUS:
                    v = torch.round(v, decimals=2)
                    p_unit = self.parameters[i].unit
                    if p_unit is not None:
                        v = str(v) + p_unit
                case ParameterType.NUMERICAL:
                    v = int(torch.round(v))
                    p_unit = self.parameters[i].unit
                    if p_unit is not None:
                        v = str(v) + p_unit
            
            f.writelines(f'{p}={v}\n')
            # logging.info(f'{p}={v}')
        
        """
        Converting categorical variables in x into the Spark configuration format.
        """    
        start = self.categorical_indices[0]
        for _ in self.categorical_indices:
            end = start + self.parameters[_].dims_required
            one_hot = x[start:end]
            cat = torch.argmax(one_hot)
            p = self.parameters[_].name
            v = self.parameters[_].items[cat]
            
            f.writelines(f'{p}={v}\n')
            # logging.info(f'{p}={v}')
            start = end
        
        f.close()
        
    def apply_and_run_configuration(self, load:bool):
        self.env.apply_configuration()
        self.env.run_configuration(load)
    
    def apply_configuration(self):
        self.env.apply_configuration()
        
    def run_configuration(self, load:bool):
        self.env.run_configuration(load)    
    
    def get_results(self) -> float:
        return self.env.get_results()

    def __call__(self, x: torch.Tensor, repeat:bool=False, load:bool=True) -> torch.Tensor:
        """
        Minimizing results
        Args:
            x (torch.Tensor): generated configuration candidates. [num, n_features]

        Returns:
            torch.Tensor: _description_
            
        """
        if repeat == BENCHMARKING_REPETITION:
            repeat = True
        else:
            repeat = False        
        
        res = []
        cnt = 0
        
        if len(x) > 1:
            logging.info(f"👌👌👌 Evaluating {len(x)} configurations 👌👌👌")
        for x_ in x:
            x_ = x_.squeeze()
        
            self.save_configuration_file(x_)           
            if repeat:
                self.apply_configuration()
                for _ in range(BENCHMARKING_REPETITION):
                    self.run_configuration(load)
                    res_ = self.get_results()
                    res.append(res_)
                    load = False
                    cnt += 1
                    logging.info(f"👌👌 [{cnt}/{len(x)}] Results:{res_:.3f} !!!!!!!!!!!!!!!")
            else:                    
                self.apply_and_run_configuration(load)
                res_ = self.get_results()
                res.append(res_) # Higher tps is better, so add the minus symbol.
                cnt += 1
                logging.info(f"👌👌 [{cnt}/{len(x)}] Results:{res_:.3f} !!!!!!!!!!!!!!!")
            
        # if len(x) > 1:
        if len(x) == BENCHMARKING_REPETITION:
            logging.info(f"👌 Results:{res}   MEAN: {mean(res)}")
            
        return torch.tensor(res)

    
# ############ VERSION FOR RECORDING THE MEAN OF REPETITION BENCHAMRKING ############
#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Minimizing results
#         Args:
#             x (torch.Tensor): generated configuration candidates. [num, n_features]

#         Returns:
#             torch.Tensor: _description_
            
#         """
#         res = []
#         for x_ in x:
#             x_ = x_.squeeze()
        
#             self.save_configuration_file(x_)
            
#             # TODO: Repeat benchmarking to minimise the impact of noise from GCP environments
#             res_ = []
#             for _ in range(BENCHMARKING_REPETITION):
#                 self.apply_and_run_configuration()
#                 res_.append(self.get_results())
#             mean_res = mean(res_)
            
#             logging.info(f"!!!!!!!!!!!!!!Results:{mean_res:.3f}!!!!!!!!!!!!!!")
#             res.append(mean_res)
            
#             # self.run_benchmark()
        
#         return torch.tensor(res)
        
#         # res = self.get_results()
#         # logging.info("########################")
#         # logging.info(f"##### res: {self.res:.2f} ######")
#         # logging.info("########################")
        
        

