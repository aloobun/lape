MODEL_NAME = "meta-llama/Llama-3.2-1B"
LANGUAGES = ["en", "hi", "bn"]

NUM_SAMPLES_PER_LANG = 2500
DATASET_NAME = "soketlabs/bhasha-wiki"
BATCH_SIZE = 8
LAPE_PERCENTILE_LOW = 1.0  #bottom 1% lang specific
LAPE_PERCENTILE_HIGH = 99.0  #top 1% for lang agnostic
ACTIVATION_PROB_THRESHOLD = 0.01 #active neurons threshold
AGNOSTIC_THRESHOLD = 0.8
POLYSEMANIC_THRESHOLD = 0.3

OUTPUT_DIR = "analysis_results"
