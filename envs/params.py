import os
from envs.gcp_info import GCP_SPARK_MASTER_ADDRESS

HOME_PATH = os.path.expanduser('~')
PROJECT_NAME = 'SparkTuning'

SPARK_CONF_INFO_CSV_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'data/Spark_3.1_45_parameters.csv')
SPARK_CONF_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'data/add-spark.conf')

DATA_FOLDER_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'data')

MASTER_ADDRESS = GCP_SPARK_MASTER_ADDRESS
MASTER_CONF_PATH = os.path.join(HOME_PATH, 'HiBench/conf')
# MASTER_BENCH_BASH = os.path.join(HOME_PATH, 'scripts/run_terasort.sh')

HIBENCH_REPORT_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'data/hibench.report')

# INCUMBENTS_RESULTS_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'results')

BOUNCE_PARAM = {
                "number_initial_points": 10,
                "initial_target_dimensionality": 5,
                "number_new_bins_on_split": 2,
                "maximum_number_evaluations": 30, # 200
                "batch_size" : 1,
                "results_dir" : "results",
                "maximum_number_evaluations_until_input_dim" : 15, # 100
                "dtype" : "float64",
                "use_scipy_lbfgs" : True
                }

TRUSTREGION_PARAM = {"length_init_discrete": 10}

GP_PARAM = {
            "lengthscale_prior_shape" : 1.5, # 3
            "lengthscale_prior_rate" : 0.1, # 6
            "outputscale_prior_shape" : 1.5, # 2
            "outputscale_prior_rate" : 0.5, # 0.15
            "noise_prior_shape" : 1.1, # 1.1
            "noise_prior_rate" : 0.05 # 2
            }

BENCHMARKING_REPETITION = 3

def print_params():
    import logging
    logging.info("📢 Information of hyperparameters")
    logging.info("================================")
    logging.info("📌Environments...")
    logging.info(f"SPARK_CONF_INFO_CSV_PATH : {SPARK_CONF_INFO_CSV_PATH}")
    logging.info(f"SPARK_CONF_PATH : {SPARK_CONF_PATH}")
    logging.info(f"MASTER_ADDRESS : {MASTER_ADDRESS}")
    logging.info(f"MASTER_CONF_PATH : {MASTER_CONF_PATH}")
    logging.info(f"HIBENCH_REPORT_PATH : {HIBENCH_REPORT_PATH}")
    
    logging.info('---------------------------')
    logging.info("📌Bounce...")
    for k, v in BOUNCE_PARAM.items():
        logging.info(f"{k} : {v}")
    
    logging.info('---------------------------')
    logging.info("📌TrustRegion...")
    for k, v in TRUSTREGION_PARAM.items():
        logging.info(f"{k} : {v}")
    
    logging.info('---------------------------')
    logging.info("📌Gaussian Process...")
    for k, v in GP_PARAM.items():
        logging.info(f"{k} : {v}")
    
    logging.info("================================")