class BColors:
    HEADER = "\033[95m"
    GREY = "\033[90m"
    WHITE = "\033[97m"
    LIGHTGREY = "\033[37m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


BOUNCE_NAME = """
    \t    ____                            
    \t   / __ )____  __  ______  ________
    \t  / __  / __ \/ / / / __ \/ ___/ _ \\
    \t / /_/ / /_/ / /_/ / / / / /__/  __/
    \t/_____/\____/\__,_/_/ /_/\___/\___/
    
    [ Bounce: Reliable High-Dimensional Bayesian Optimization     ] 
    [ Algorithm for Combinatorial and Mixed Spaces ]
    """

BOUNCE_NAME = f"{BColors.LIGHTGREY}{BOUNCE_NAME}{BColors.ENDC}"

RANDOM_NAME = """
\t     ____                  __ 
\t    / __ \____ _____  ____/ /___  ____ ___ 
\t   / /_/ / __ `/ __ \\/ __  / __ \/ __ `__ \\
\t  / _  _/ /_/ / / / / /_/ / /_/ / / / / / /
\t /_/ |_|\__,_/_/ /_/\__,_/\____/_/ /_/ /_/ 

  [ Random Search mode ]
"""

NSBO_NAME = """
\t    _   _______ ____  ____
\t   / | / / ___// __ )/ __ \\
\t  /  |/ /\__ \/ __  / / / /
\t / /|  /___/ / /_/ / /_/ /
\t/_/ |_/_____/_____/\____/


   [ NSBO: Nested Subspace-based Bayesian Optimization ]
"""

HESBO_NAME = """
\t    __  __          ____  ____
\t   / / / /__  _____/ __ )/ __ \\
\t  / /_/ / _ \\/  __/ __  / / / /
\t / __  /  __/_\\ \\/ /_/ / /_/ /
\t/_/ /_/\\___/____/_____/\____/

    [ HesBO: Using HesBO leveraged to SMAC --> LlamaTune ]
"""