import os
from envs.gcp_info import GCP_SPARK_MASTER_ADDRESS

HOME_PATH = os.path.expanduser('~')
PROJECT_NAME = 'SparkTuning'

SPARK_CONF_INFO_CSV_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'data/Spark_3.1_45_parameters.csv')
SPARK_CONF_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'data/add-spark.conf')

MASTER_ADDRESS = GCP_SPARK_MASTER_ADDRESS
MASTER_CONF_PATH = os.path.join(HOME_PATH, 'HiBench/conf')
# MASTER_BENCH_BASH = os.path.join(HOME_PATH, 'scripts/run_terasort.sh')

# HIBENCH_REPORT_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'data/hibench.report')

# INCUMBENTS_RESULTS_PATH = os.path.join(HOME_PATH, PROJECT_NAME, 'results')