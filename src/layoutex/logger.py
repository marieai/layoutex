import logging

def configure_logger(filename="app.log", level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    logging.basicConfig(level=level, format=format)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(level)