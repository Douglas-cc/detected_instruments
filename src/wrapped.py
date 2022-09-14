import os
import pickle 
import numpy as np
import pandas as pd


class Wrapped:
    def __init__(self, row, processed, files):
        self.directory_row = row
        self.directory_processed = processed
        self.directory_files = files
    
    
    def create_directory(self, folders, directory):
        for folder in folders:
            print(f'Creating folder: { folder } in { directory }')
            os.makedirs(f'{ directory }{ folder }', exist_ok=True)
    
    
    def save_data(self, name_out, variable):
        f = open(self.directory_row+name_out, 'wb')
        pickle.dump(variable, f)
        f.close()    
        
    
    def load_data(self, file):
        f = open(self.directory_row+file, 'rb')  
        aux = pickle.load(f)
        f.close()
        return aux  