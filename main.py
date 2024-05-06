import argparse
import logging
import time
import os

# import gin

from bounce.bounce import Bounce
from bounce.util.printing import BColors, BOUNCE_NAME, RANDOM_NAME, INCPP_NAME, HESBO_NAME
from bounce.spark_benchmark import SparkTuning
from random_search.search import RandomSearch
from random_search.benchmarks import SparkBench
from incpp.incpp import incPP
from others.optimizers import Baselines
from others.benchmarks import Benchmark

from envs.utils import get_logger
from envs.spark import SparkEnv

from envs.params import print_params
from envs.params import BOUNCE_PARAM as bp

logger = get_logger('logs')
os.system('clear')
DEBUGGING_MODE = False

def main():
    parser = argparse.ArgumentParser(
        prog=BOUNCE_NAME,
        description="Bounce: Reliable High-Dimensional Bayesian Optimization Algorithm for Combinatorial and Mixed Spaces",
        epilog="For more information, please contact the author.",
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default='bounce',
        help='bounce, random, ...'
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["aggregation", "join", "scan", "wordcount", "terasort", "bayes", "kmeans", "pagerank", "svm", "nweight"],
        default="join"
    )
    parser.add_argument(
        "--workload_size",
        type=str,
        choices=["tiny", "small", "large"],
        default="large"
    )
    # =======For evaluating modules developed currently=======    
    parser.add_argument(
        "--neighbor",
        type=float,
        default=0.03,
        help='[BO-PP] adjusting a degree of distances from neighbors'
    )
    parser.add_argument(
        "--wo_bopp",
        action='store_false',
        help='[BO-PP] If you want to run without a BO-PP module, trigger this'
    )
    parser.add_argument(
        "--bopp_ratio",
        type=float,
        default=1.0,
        help='[BO-PP] Adjusting using best sample ratio for a BO-PP'
    )
    # parser.add_argument(
    #     "--gp",
    #     type=str,
    #     default='fixednoisegp',
    #     choices=['fixednoisegp', 'singletaskgp'],
    #     help='[Noise-GP] Choosing the kind of GP class whether inseting noise variances or not'
    # )
    parser.add_argument(
        "--bin",
        type=int,
        default=2,
        help='[Bounce] adjusting the number of new bins on splitting'
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=10,
        help='[Bounce] adjusting init sampling sizes'
    )
    parser.add_argument(
        "--target_dim",
        type=int,
        default=bp['initial_target_dimensionality'],
        help='[Bounce&HesBO&LlamaTune] adjusting init target dimensionality'
    )    
    parser.add_argument(
        "--max_eval",
        type=int,
        default=bp["maximum_number_evaluations"],
        help='[Bounce] adjusting init sampling sizes'
    )
    parser.add_argument(
        "--max_eval_until_input",
        type=int,
        default=bp["maximum_number_evaluations_until_input_dim"],
        help='[Bounce] adjusting init sampling sizes until reaching input dimensions'
    )
    # parser.add_argument(
    #     "--noise_free",
    #     action='store_true',
    #     help='[Noise] If you want to run benchmarking in a noise-free experiment, trigger this'
    # )
    parser.add_argument(
        "--noise_mode",
        type=int,
        choices=[1, 2, 3],
        help='[Noise] Choose noise mode, \
                1: a noisy observation mode, \
                2: a noise-free mode w repeated evaluating, \
                3: a noise-free mode w repeated experiments'
    )
    parser.add_argument(
        "--debugging",
        action='store_true',
        help='[DEBUGGING] If you want to debug the entire code without running benchmarking, trigger this'
    )
    parser.add_argument(
        "--q_factor",
        type=int,
        default=None,
        help='[LlamaTune] adjusting quantization factor (configuration space bucketization)'
    )   
    # ========================================================
    
    args = parser.parse_args()
    
    global DEBUGGING_MODE
    DEBUGGING_MODE = True if args.debugging else False
    
    
    logging.basicConfig(
        level=logging.INFO,
        format=f"{BColors.LIGHTGREY} %(levelname)s:%(asctime)s - (%(filename)s:%(lineno)d) - %(message)s {BColors.ENDC}",
    )

    if args.method == 'bounce':
        logging.info(BOUNCE_NAME)
    elif args.method == 'random':
        logging.info(RANDOM_NAME)
    elif args.method == 'incpp':
        logging.info(INCPP_NAME)
    elif args.method == 'hesbo':
        logging.info(HESBO_NAME)            

    # parser.add_argument(
    #     "--gin-files",
    #     type=str,
    #     nargs="+",
    #     default=["configs/my.gin"],
    #     help="Path to the config file",
    # )
    # parser.add_argument(
    #     "--gin-bindings",
    #     type=str,
    #     nargs="+",
    #     default=[],
    # )

    print_params()

    ## print parser info
    logger.info("📢 Argument information ")
    logger.info("*************************************")
    for i in vars(args):
        logger.info(f'{i}: {vars(args)[i]}')
    logger.info("*************************************")
    
    
    # gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)

    env = None
    
    match args.method:
        case "bounce":
            env = SparkEnv(
                workload=args.workload,
                workload_size=args.workload_size,
                debugging=args.debugging
                )
            benchmark = SparkTuning(env=env)
            tuner = Bounce(benchmark=benchmark)
        case "random":            
            benchmark = SparkBench(
                workload=args.workload,
                workload_size=args.workload_size,
                debugging=args.debugging
                )
            tuner = RandomSearch(benchmark=benchmark,
                                 maximum_number_evaluations=args.max_eval)
        case "incpp":
            env = SparkEnv(
                workload=args.workload,
                workload_size=args.workload_size,
                debugging=args.debugging
                )
            benchmark = SparkTuning(env=env)
            tuner = incPP(
                benchmark=benchmark, 
                neighbor_distance=args.neighbor, 
                pseudo_point=args.wo_bopp,
                pseudo_point_ratio=args.bopp_ratio,
                initial_target_dimensionality=args.target_dim,
                bin=args.bin,
                n_init=args.n_init,
                max_eval=args.max_eval,
                max_eval_until_input=args.max_eval_until_input,
                noise_mode=args.noise_mode,
            #   gp_mode=args.gp
                )
        case "hesbo":
            benchmark = Benchmark(
                workload=args.workload,
                workload_size=args.workload_size,
                debugging=args.debugging,
                embed_adapter_alias=args.method,
                target_dim=args.target_dim,
                quantization_factor=args.q_factor,
                )
            tuner = Baselines(
                method=args.method,
                benchmark=benchmark
                )
        case _:
            assert False, "The method is not defined.. Choose in [bounce, random]"
    
    then = time.time()
    tuner.run()
    
    now = time.time()
    logger.info(f"Total time: {now - then:.2f} seconds")
    
    if env is not None:
        env.clear_spark_storage()
        env.stop_dataproc()
    else: 
        # the case for random search module
        benchmark.clear_spark_storage()
        benchmark.stop_dataproc()
    


if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception("ERROR!!")
               
        if DEBUGGING_MODE:
            logging.info("Skipping stop GCP instance")    
        else:
            logging.info("[Google Cloud Platform|Dataproc] ⛔ Stop Spark instances")
            from envs.params import GCP_DATAPROC_STOP_COMMAND
            os.system(GCP_DATAPROC_STOP_COMMAND)
    else:
        logger.handlers.clear()

