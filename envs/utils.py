import datetime
import os, logging
import pandas as pd

def get_filename(PATH, head, tail):
    i = 0
    today = datetime.datetime.now()
    today = today.strftime('%Y%m%d')
    os.makedirs(os.path.join(PATH, today), exist_ok=True)
    # if not os.path.exists(os.path.join(PATH, today)):
    #     os.mkdir(os.path.join(PATH, today))
    name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    while os.path.exists(os.path.join(PATH, name)):
        i += 1
        name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    return name

def get_foldername(PATH):
    i = 0
    today = datetime.datetime.now()
    today = today.strftime('%Y%m%d')
    os.makedirs(PATH, exist_ok=True)
    # if not os.path.exists(PATH):
    #     os.mkdir(PATH)
    folder_name = os.path.join(PATH, today + '-%02d'%i)
    while os.path.exists(folder_name):
        i += 1
        folder_name = os.path.join(PATH, today + '-%02d'%i)
    return folder_name
        
        
def get_logger(log_path='./logs'):
    os.makedirs(log_path, exist_ok=True)

    logger = logging.getLogger()

    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('%(asctime)s[%(levelname)s] %(filename)s:%(lineno)s  %(message)s', date_format)
    name = get_filename(log_path, 'log', '.log')

    fileHandler = logging.FileHandler(os.path.join(log_path, name))
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path, name)))
    # return logger, os.path.join(log_path, name)
    return logger

def print_configuration_from_dict(x: dict[str,float]):
    assert isinstance(x, dict), "x should be dictionary {str: float}"
    data_info = pd.read_csv("data/Spark_3.1_45_parameters.csv", index_col=0).to_dict(orient='index')

    for p in x:
        p_info = data_info[p]
        v = x[p]
        unit = p_info['unit']
        
        match p_info['type']:
            case 'binary':
                items = p_info['range'].split(',')
                v = items[int(v)]
                print(f'{p}={v}')
            case 'categorical':
                items = p_info['range'].split(',')
                v = items[int(v)]
                print(f'{p}={v}')
            case 'numerical':
                print(f'{p}={int(v)}') if pd.isna(unit) else print(f'{p}={int(v)}{unit}')
            case 'continuous':
                print(f'{p}={v:.2f}')

def logging_configuration_from_dict(x: dict[str,float]):
    assert isinstance(x, dict), "x should be dictionary {str: float}"
    data_info = pd.read_csv("data/Spark_3.1_45_parameters.csv", index_col=0).to_dict(orient='index')

    for p in x:
        p_info = data_info[p]
        v = x[p]
        unit = p_info['unit']
        
        match p_info['type']:
            case 'binary':
                items = p_info['range'].split(',')
                v = items[int(v)]
                logging.info(f'{p}={v}')
            case 'categorical':
                items = p_info['range'].split(',')
                v = items[int(v)]
                logging.info(f'{p}={v}')
            case 'numerical':
                logging.info(f'{p}={int(v)}') if pd.isna(unit) else print(f'{p}={int(v)}{unit}')
            case 'continuous':
                logging.info(f'{p}={v:.2f}')