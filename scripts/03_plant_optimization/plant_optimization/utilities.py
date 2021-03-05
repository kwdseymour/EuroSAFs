import os
import logging

def create_logger(scratch_path,name,file):
    # Add a logger
    file_string = os.path.basename(file).split('.')[0]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(levelname)s: %(message)s','%Y-%m-%d %H:%M:%S')
    if not os.path.isdir(os.path.join(scratch_path,'logs')):
        os.mkdir(os.path.join(scratch_path,'logs'))
    file_handler1 = logging.FileHandler(os.path.join(scratch_path,'logs',f'{file_string}_persistent.log'))
    file_handler1.setLevel(logging.INFO)
    file_handler1.setFormatter(formatter)
    file_handler2 = logging.FileHandler(os.path.join(scratch_path,'logs',f'{file_string}.log'),mode='w')
    file_handler2.setLevel(logging.INFO)
    file_handler2.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler1)
    logger.addHandler(file_handler2)
    logger.addHandler(stream_handler)
    logger.propogate = False
    return logger