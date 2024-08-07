# NoRTune
We introduce NoRTune, a resource-efficient and reliable configuration tuning framework for Spark that leverages subspace-based Bayesian optimization and a noise-robust acquisition function. NoRTune optimizes Spark configuration by effectively reducing high-dimensional parameters without requiring time-consuming and resource-intensive determination of the target dimensionality. Moreover, it is able to select an optimal configuration with reliable performance by reducing the impact of noise.

* The base of this code is [here](https://github.com/LeoIV/Bounce).


## Installation

Bounce uses `poetry` for dependency management.
`Bounce` requires `python>=3.10`. To install `poetry`, see [here](https://python-poetry.org/docs/#installation).
To install the dependencies, run

```bash
poetry install
pip install configspace
```
