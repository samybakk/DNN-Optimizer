import logging

def setup_logger(log_dir):
    logger = logging.getLogger(f'{__name__}_{log_dir}')
    logger.setLevel(logging.INFO)
    
    # Create a file handler
    file_handler = logging.FileHandler(log_dir+'/log.txt')
    file_handler.setLevel(logging.INFO)
    
    # Create a log message format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    print(f'Logger is set up with path : {log_dir}/log.txt')
    
    return logger