import argparse
import logging
import time
import os

import gin

from bounce.bounce import Bounce
from bounce.util.printing import BColors, BOUNCE_NAME
from bounce.spark_benchmark import SparkTuning
from random_search.search import RandomSearch
from random_search.benchmarks import SparkBench

from envs.utils import get_logger

logger = get_logger()
os.system('clear')

def main():
    logging.basicConfig(
        level=logging.INFO,
        format=f"{BColors.LIGHTGREY} %(levelname)s:%(asctime)s - (%(filename)s:%(lineno)d) - %(message)s {BColors.ENDC}",
    )

    logging.info(BOUNCE_NAME)

    then = time.time()
    parser = argparse.ArgumentParser(
        prog=BOUNCE_NAME,
        description="Bounce: Reliable High-Dimensional Bayesian Optimization Algorithm for Combinatorial and Mixed Spaces",
        epilog="For more information, please contact the author.",
    )

    parser.add_argument(
        "--gin-files",
        type=str,
        nargs="+",
        default=["configs/my.gin"],
        help="Path to the config file",
    )
    parser.add_argument(
        "--gin-bindings",
        type=str,
        nargs="+",
        default=[],
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
        choices=["aggregation", "join", "scan", "wordcount", "terasort", "bayes", "kmeans", "pagerank"],
        default="join"
    )


    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)

    match args.method:
        case "bounce":
            benchmark = SparkTuning(workload=args.workload)
            tuner = Bounce(benchmark=benchmark)
        case "random":            
            benchmark = SparkBench(workload=args.workload)
            tuner = RandomSearch(benchmark=benchmark)
        case _:
            assert False, "The method is not defined.. Choose in [bounce, random]"
    
    tuner.run()
    # bounce = Bounce()
    
    # bounce.run()

    gin.clear_config()
    now = time.time()
    logger.info(f"Total time: {now - then:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception("ERROR!!")
    else:
        logger.handlers.clear()

