import os
import sys

# Set the path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
# Set the paths to the data files
INDEX_URL = "https://makerlab.illinois.edu/"

DATA_MODEL_DIR = os.path.join(PROJECT_ROOT, 'data')
# Set the paths to the model files
T5_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 't5-base')
T5_CONFIG_FILE = os.path.join(T5_MODEL_DIR, 'config.json')
# T5_MODEL_FILE = os.path.join(T5_MODEL_DIR, 'pytorch_model.bin')
T5_VOCAB_FILE = os.path.join(T5_MODEL_DIR, 'vocab.bin')

GPT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'gpt3')
GPT_CONFIG_FILE = os.path.join(GPT_MODEL_DIR, 'config.json')
# GPT_MODEL_FILE = os.path.join(GPT_MODEL_DIR, 'pytorch_model.bin')
