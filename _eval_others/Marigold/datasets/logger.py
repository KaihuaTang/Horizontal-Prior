import os
import re
import numpy as np
import logging
from datetime import datetime

class CustomLogger(object):
    def __init__(self, name, output_folder, rank=0):
        self.rank = rank
        self.name = name
        self.output_folder = output_folder
        self.logger_file = os.path.join(self.output_folder, self.get_time() + "_" + name + ".txt")

        os.makedirs(self.output_folder, exist_ok=True)
        with open(self.logger_file, 'a') as file:
            file.write(f"Init Logger at rank {self.rank}" + '\n')

    def get_time(self):
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute
        time_string = f"{year}_{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
        return time_string

    def info(self, message, print_rank=0):
        with open(self.logger_file, 'a') as file:
            file.write(f"Rank {self.rank}: {message}" + '\n')
        if self.rank == print_rank:
            print(message)

def init_log(name, file_path, rank=0):
    logger = CustomLogger(name, file_path, rank=rank)
    return logger
