import os
import random
import string
from config import Config

class BaseController:
    
    def __init__(self):
        # Bind absolute directory paths using the centralized system configurations
        self.base_dir = Config.BASE_DIR
        self.files_dir = Config.FILES_DIR
        
        # Programmatically guarantee the permanent archival storage folder exists
        if not os.path.exists(self.files_dir):
            os.makedirs(self.files_dir)
            
    def generate_random_string(self, length: int = 12):
        # Construct isolated unique secure identifier strings for file hashing
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))