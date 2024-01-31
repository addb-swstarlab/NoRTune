import argparse
import logging
import time
import os

import gin

from bounce.bounce import Bounce
from bounce.util.printing import BColors, BOUNCE_NAME

from envs.utils import get_logger

logger = get_logger()
os.system('clear')

if __name__ == "__main__":

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

    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)

    bounce = Bounce()
    
    bounce.run()

    gin.clear_config()
    now = time.time()
    logging.info(f"Total time: {now - then:.2f} seconds")
