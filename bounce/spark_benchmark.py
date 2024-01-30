from bounce.benchmarks import Benchmark
from bounce.util.benchmark import Parameter, ParameterType

import gin
import pandas as pd
import torch
import logging

@gin.configurable
class SparkTuning(Benchmark):
    def __init__(
        self,
        n_features: int = 45,
        csv_path: str = '/path/to/read/csv',
        config_path: str = '/path/to/save/configuration'
    ):
        self.n_features = n_features
        self.config_path = config_path
        csv_data = pd.read_csv(csv_path, index_col=0)
        self.dict_data = csv_data.to_dict(orient='index')
        
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
                items=None if self.dict_data[k]['item'] != self.dict_data[k]['item'] else self.dict_data[k]['item'].split(',')
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
        
        # parameters = [
        #     Parameter(
        #         name=k,
        #         type=getattr(ParameterType, dict_data[k]['type'].upper()),
        #         lower_bound=dict_data[k]['min'],
        #         upper_bound=dict_data[k]['max']
        #     )
        #     for k in dict_data.keys()
        # ]
        super().__init__(parameters=parameters, noise_std=None)

        self.flip = False
        # self.train_x, self.train_y, self.test_x, self.test_y = load_uci_data(
        #     n_features=n_features
        # )

        # self.flip_tensor = torch.tensor(RandomState(0).choice([0, 1], n_features))
        """
        the tensor used to flip the binary parameters
        """
    
    def save_configuration_file(self, x: torch.Tensor):
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
                    v = int(v)
                    v = self.parameters[i].items[v]
                case ParameterType.CONTINUOUS:
                    v = torch.round(v, decimals=2)
                    p_unit = self.parameters[i].unit
                    if p_unit is not None:
                        v = str(v) + p_unit
                case ParameterType.NUMERICAL:
                    v = int(v)
                    p_unit = self.parameters[i].unit
                    if p_unit is not None:
                        v = str(v) + p_unit
            
            f.writelines(f'{p}={v}\n')
            logging.info(f'{p}={v}')
        
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
            logging.info(f'{p}={v}')
            start = end
        
        f.close()
         
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        # TODO: 1. Converting x with a Tensor type into the Spark configuration format.
        # Complete!
        self.save_configuration_file(x)
        assert False
        
        # TODO: 2. Transporting the created spark configuration to Spark master node to apply the configuration setting.
        # TODO: 3. Running HiBench to benchmark Spark with the configuration.
        # TODO: 4. Receiving the performance results.
        return 0
        

