import argparse
import logging
import time
import os

# import gin

from bounce.bounce import Bounce
from bounce.util.printing import BColors, BOUNCE_NAME, RANDOM_NAME, NSBO_NAME, HESBO_NAME
from bounce.postgres_benchmark import PostgresTuning
from random_search.search import RandomSearch
from random_search.benchmarks import PostgresBench
from nsbo.nsbo import NSBO
from others.optimizers import Baselines
from others.benchmarks import Benchmark, PostgresBenchmark

from envs.utils import get_logger
from envs.postgres import PostgresEnv

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
        "--model_name",
        type=str,
        default='not named',
        help='Define a model name for this experiment'
    )
    parser.add_argument(
        "--optimizer_method",
        type=str,
        default='bounce',
        choices=['bounce', 'random', 'nsbo', 'bo', 'smac'],
        help='bounce, random, ...'
    )
    parser.add_argument(
        "--embedding_method",
        type=str,
        default='bounce',
        choices=['hesbo', 'rembo', 'none'],
        help='bounce, random, ...'
    )    
    parser.add_argument(
        "--workload",
        type=str,
        choices=['ycsb-a', 'ycsb-b'],
        default="join"
    )
    # parser.add_argument(
    #     "--workload_size",
    #     type=str,
    #     choices=["tiny", "small", "large", "huge", "gigantic"],
    #     default="large"
    # )
    parser.add_argument(
        "--bin",
        type=int,
        default=1,
        help='[Bounce] adjusting the number of new bins on splitting'
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=5,
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
    # parser.add_argument(
    #     "--noise_mode",
    #     type=int,
    #     choices=[1, 2, 3, 4, 5],
    #     default=5,
    #     help='[Noise] Choose noise mode, \
    #             1: a noisy observation mode, \
    #             2: a noise-free mode w repeated evaluating, \
    #             3: a noise-free mode w repeated experiments, \
    #             4: an adaptive noisy observation mode. \
    #             5: a noisy observation mode using mean'
    # )
    parser.add_argument(
        "--noise_threshold",
        type=float,
        default= 1,
        help='[Noise] Define std threshold to adjust a degree of noise'
    )
    parser.add_argument(
        "--acquisition",
        type=str,
        default='ei',
        choices=['ei', 'aei'],
        help='[Noise] Define which acquisition function is used.'
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
    parser.add_argument(
        "--is_tps",
        action='store_true',
        help='[Metrics] If tuning tps, trigger this.'
    )
    
    
    args = parser.parse_args()
    
    global DEBUGGING_MODE
    DEBUGGING_MODE = True if args.debugging else False
    
    
    logging.basicConfig(
        level=logging.INFO,
        format=f"{BColors.LIGHTGREY} %(levelname)s:%(asctime)s - (%(filename)s:%(lineno)d) - %(message)s {BColors.ENDC}",
    )

    if args.optimizer_method == 'bounce':
        logging.info(BOUNCE_NAME)
    elif args.optimizer_method == 'random':
        logging.info(RANDOM_NAME)
    elif args.optimizer_method == 'nsbo':
        logging.info(NSBO_NAME)
    else:
        logging.info("🟥🟧🟨🟩🟦🟪🟦🟩🟨🟧🟥")
        logging.info(args.model_name)
        logging.info("🟥🟧🟨🟩🟦🟪🟦🟩🟨🟧🟥")
    # elif args.method == 'hesbo':
    #     logging.info(HESBO_NAME)            

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
    
    match args.optimizer_method:
        case "bounce":
            env = PostgresEnv(
                workload=args.workload,
                # workload_size=args.workload_size,
                debugging=args.debugging
                )
            benchmark = PostgresTuning(env=env)
            tuner = Bounce(benchmark=benchmark)
        case "random":            
            benchmark = PostgresBench(
                workload=args.workload,
                # workload_size=args.workload_size,
                debugging=args.debugging
                )
            tuner = RandomSearch(benchmark=benchmark,
                                 maximum_number_evaluations=args.max_eval,
                                 is_tps=True)
        case "nsbo":
            env = PostgresEnv(
                workload=args.workload,
                # workload_size=args.workload_size,
                debugging=args.debugging
                )
            benchmark = PostgresTuning(env=env)
            tuner = NSBO(
                benchmark=benchmark, 
                initial_target_dimensionality=args.target_dim,
                bin=args.bin,
                n_init=args.n_init,
                max_eval=args.max_eval,
                max_eval_until_input=args.max_eval_until_input,
                # noise_mode=args.noise_mode,
                noise_threshold=args.noise_threshold,
                acquisition=args.acquisition,
            #   gp_mode=args.gp
                )
        case "smac":
            benchmark = PostgresBenchmark(
                workload=args.workload,
                # workload_size=args.workload_size,
                debugging=args.debugging,
                embed_adapter_alias=args.embedding_method,
                target_dim=args.target_dim,
                quantization_factor=args.q_factor,
                )
            tuner = Baselines(
                optimizer_method=args.optimizer_method,
                embedding_method=args.embedding_method,
                benchmark=benchmark,
                is_tps=True,
                )
        case "bo":
            benchmark = PostgresBenchmark(workload=args.workload,
                                #   workload_size=args.workload_size,
                                  debugging=args.debugging,
                                  embed_adapter_alias=args.embedding_method,
                                  target_dim=args.target_dim,
                                  quantization_factor=args.q_factor,
                                  )
            tuner = Baselines(
                optimizer_method=args.optimizer_method,
                embedding_method=args.embedding_method,
                benchmark=benchmark,
                acquisition_function=args.acquisition,
                is_tps=True,
                )
        case _:
            assert False, "The method is not defined.. Choose in [bounce, random]"
    
    then = time.time()
    tuner.run()
    
    now = time.time()
    logger.info(f"Total time: {now - then:.2f} seconds")
    
    # if env is not None:
    #     env.clear_spark_storage()
    #     env.stop_dataproc()
    # else: 
    #     # the case for random search module
    #     benchmark.clear_spark_storage()
    #     benchmark.stop_dataproc()
    


if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception("ERROR!!")
               
        # if DEBUGGING_MODE:
        #     logging.info("Skipping stop GCP instance")    
        # else:
            # logging.info("[Google Cloud Platform|Dataproc] ⛔ Stop Spark instances")
            # from envs.params import GCP_DATAPROC_STOP_COMMAND
            # os.system(GCP_DATAPROC_STOP_COMMAND)
    else:
        logger.handlers.clear()
