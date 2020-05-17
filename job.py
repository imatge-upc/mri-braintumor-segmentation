#!/home/usuaris/imatge/laura.mora/tfmenv/bin/python
import sys
import os

# Set up env variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


sys.path.append(os.getcwd())
os.system('python src/main.py resources/config.ini')