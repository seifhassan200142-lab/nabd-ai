import os
import re
from .BaseController import BaseController

class DataController(BaseController):
    
    def __init__(self):
        # Inherit infrastructural tools and tracking parameters from BaseController
        super().__init__()

    def get_clean_file_name(self, orig_file_name: str):
        # Sanitize incoming document names to extract raw alphanumeric and dot attributes
        cleaned_file_name = re.sub(r'[^\w.]', '', orig_file_name.strip())
        cleaned_file_name = cleaned_file_name.replace(" ", "_")
        return cleaned_file_name

    def generate_unique_filepath(self, orig_file_name: str, project_id: str):
        # Isolate patient assets via localized secure subdirectories
        random_key = self.generate_random_string()
        project_path = os.path.join(self.files_dir, project_id)
        
        if not os.path.exists(project_path):
            os.makedirs(project_path)

        # Build clean dynamic targeted absolute physical disk address strings
        cleaned_file_name = self.get_clean_file_name(orig_file_name=orig_file_name)
        final_file_name = f"{random_key}_{cleaned_file_name}"
        new_file_path = os.path.join(project_path, final_file_name)

        return new_file_path, final_file_name
    
    