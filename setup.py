import os 
from pathlib import Path
import logging

project_name="automate"

files=[
    f"{project_name}/frontend/pages/prediction.py",
    f"{project_name}/frontend/main.py",
    f"{project_name}/frontend/pages/classification.py",
    f"{project_name}/backend/main.py",
    f"{project_name}/backend/models.py",
    "requirements.txt"
]

logging.basicConfig(level=logging.INFO)

for file in files:
    file_path=Path(file)
    file_dir,file_name=os.path.split(file_path)

    if file_dir!="":
        os.makedirs(file_dir,exist_ok=True)
        logging.info("directry created")
    
    if not os.path.exists(file_path) or os.path.getsize(file_path)==0:
        with open(file_path,"w") as f:
            pass
        logging.info("file created")
    else:
        logging.info("file esist")