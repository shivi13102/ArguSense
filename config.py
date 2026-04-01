import os

# --- General Configuration ---
RANDOM_SEED = 42
TEST_SIZE = 0.2
STRATIFY = True

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "DATASETS")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Sub-directories
SARCASM_MODELS_DIR = os.path.join(MODELS_DIR, "sarcasm")
ARGUMENT_MODELS_DIR = os.path.join(MODELS_DIR, "argument")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# --- Dataset Configuration ---
SARCASM_DATA_PATH = os.path.join(DATASETS_DIR, "News Headlines Dataset For Sarcasm Detection", "Sarcasm_Headlines_Dataset_v2.json")
ARGUMENT_DATA_PATH = os.path.join(DATASETS_DIR, "feedback-prize-effectiveness", "train.csv")

# Training refinement
# v3.0 Upgrade: Default to None for full data training
# Set to an integer (e.g., 15000) for faster debug cycles
ARGUMENT_SAMPLE_SIZE = None

# --- Feature Engineering (Dual TF-IDF) ---
USE_WORD_TFIDF = True
USE_CHAR_TFIDF = True
USE_HANDCRAFTED_FEATURES = False # Optional linguistic stats

# Word-Level Settings
MAX_FEATURES_WORD = 50000
NGRAM_RANGE_WORD = (1, 2)

# Character-Level Settings (char_wb)
MAX_FEATURES_CHAR = 30000
NGRAM_RANGE_CHAR = (3, 5)

# Shared Vectorizer Settings
MIN_DF = 2
MAX_DF = 0.95
SUBLINEAR_TF = True

# Advanced Features
CLASS_WEIGHT_BALANCED = True

# --- Hyperparameter Tuning (Grid Config) ---
TUNING_GRIDS = {
    'SGD_ALPHA': [0.0001, 0.001, 0.01],
    'LOGREG_C': [0.1, 1.0, 10.0],
    'LINEARSVC_C': [0.1, 1.0, 10.0],
    'NB_ALPHA': [0.01, 0.1, 1.0]
}
PRIMARY_METRIC = 'weighted_f1'

# --- Label Mappings ---
SARCASM_MAPPING = {0: "Not Sarcastic", 1: "Sarcastic"}
ARGUMENT_MAPPING = {0: "Ineffective", 1: "Adequate", 2: "Effective"}

# --- Streamlit Constants ---
APP_TITLE = "ArguSense v3.0"
APP_SUBTITLE = "Advanced Sarcasm-Aware Argument Benchmarking"
