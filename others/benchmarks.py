import logging
from ConfigSpace import Configuration, ConfigurationSpace
from random_search.benchmarks import SparkBench
from others.adapters.low_embeddings import LinearEmbeddingConfigSpace

class Benchmark(SparkBench):
    def __init__(
        self,
        embed_adapter_alias : str,
        target_dim: int,
        workload: str = None,
        workload_size: str = None,
        alter: bool = True,
        debugging: bool = False,
        quantization_factor: int = None,
    ):
        self.embed_adapter_alias = embed_adapter_alias
        self.target_dim = target_dim
        self._quantization_factor = quantization_factor
        
        assert self.embed_adapter_alias in ['rembo', 'hesbo', 'ddpg', 'none'], "embed_adapter_alias should be defined to 'rembo', 'hesbo', or 'ddpg'."
        
        super().__init__(workload=workload, workload_size=workload_size, alter=alter, debugging=debugging)

        self.input_space: ConfigurationSpace = self.spark_cs
        
        if self.embed_adapter_alias in ['rembo', 'hesbo']:
            self.input_space = self._get_embedding_space(self.embed_adapter_alias)
        
        
    def _get_embedding_space(self, method:str):
        logging.info(f"📢 The embedding type is {method}.")
        self.embedding_adapter = LinearEmbeddingConfigSpace.create(
            adaptee=self.spark_cs,
            target_dim=self.target_dim, # x_low_dim, subspace dim
            method=self.embed_adapter_alias,
            seed=0,
            max_num_values=self._quantization_factor,
        )
        
        return self.embedding_adapter.target
        
    
    def evaluate(self, sample: Configuration, load: bool, seed: int = 0) -> float:
        if self.embed_adapter_alias in ['rembo', 'hesbo']:
            sample = self.embedding_adapter.unproject_point(sample)
        
        self.save_configuration_file(sample)
        self.apply_and_run_configuration(load)

        res = self.get_results()
        return res