import logging
from config import (
    MODEL_NAME, LANGUAGES, NUM_SAMPLES_PER_LANG, DATASET_NAME, BATCH_SIZE,
    LAPE_PERCENTILE_LOW, LAPE_PERCENTILE_HIGH, ACTIVATION_PROB_THRESHOLD,
    AGNOSTIC_THRESHOLD, POLYSEMANIC_THRESHOLD, OUTPUT_DIR
)
from lape import LAPE

def main():
    analyzer = LAPE(MODEL_NAME, LANGUAGES)
    texts = analyzer.load_data(NUM_SAMPLES_PER_LANG, DATASET_NAME)
    
    analyzer.get_activation_probabilities(texts, batch_size=BATCH_SIZE, activation_threshold=0.0)
    distributions = analyzer.analyze_activation_distributions()
    
    print("\nActivation Probability Statistics:")
    print(f"  1st percentile: {distributions['activation_probabilities']['statistics']['percentiles']['1%']:.6f}")
    print(f"  5th percentile: {distributions['activation_probabilities']['statistics']['percentiles']['5%']:.6f}")
    print(f"  10th percentile: {distributions['activation_probabilities']['statistics']['percentiles']['10%']:.6f}")
    print(f"  25th percentile: {distributions['activation_probabilities']['statistics']['percentiles']['25%']:.6f}")
    print(f"  Median: {distributions['activation_probabilities']['statistics']['percentiles']['50%']:.6f}")
    print(f"  75th percentile: {distributions['activation_probabilities']['statistics']['percentiles']['75%']:.6f}")
    print(f"  95th percentile: {distributions['activation_probabilities']['statistics']['percentiles']['95%']:.6f}")
    print(f"  99th percentile: {distributions['activation_probabilities']['statistics']['percentiles']['99%']:.6f}")
    
    print("\nLAPE Score Statistics:")
    print(f"  1st percentile: {distributions['lape_scores']['statistics']['percentiles']['1%']:.6f}")
    print(f"  5th percentile: {distributions['lape_scores']['statistics']['percentiles']['5%']:.6f}")
    print(f"  10th percentile: {distributions['lape_scores']['statistics']['percentiles']['10%']:.6f}")
    print(f"  25th percentile: {distributions['lape_scores']['statistics']['percentiles']['25%']:.6f}")
    print(f"  Median: {distributions['lape_scores']['statistics']['percentiles']['50%']:.6f}")
    print(f"  75th percentile: {distributions['lape_scores']['statistics']['percentiles']['75%']:.6f}")
    print(f"  95th percentile: {distributions['lape_scores']['statistics']['percentiles']['95%']:.6f}")
    print(f"  99th percentile: {distributions['lape_scores']['statistics']['percentiles']['99%']:.6f}")
    
    analyzer.get_activation_probabilities(
        texts, 
        batch_size=BATCH_SIZE, 
        activation_threshold=ACTIVATION_PROB_THRESHOLD
    )
    
    analyzer.classify_neurons(
        lape_percentile_low=LAPE_PERCENTILE_LOW,
        lape_percentile_high=LAPE_PERCENTILE_HIGH,
        activation_prob_threshold=ACTIVATION_PROB_THRESHOLD,
        agnostic_threshold=AGNOSTIC_THRESHOLD,
        polysemantic_threshold=POLYSEMANIC_THRESHOLD
    )
    
    analyzer.analyze_polysemantic_neurons(activation_prob_threshold=ACTIVATION_PROB_THRESHOLD)
    
    polysemantic_neurons = analyzer.neuron_types.get("polysemantic", [])
    sample_size = min(100, len(polysemantic_neurons))  # Analyze up to 100 neurons
    sampled_neurons = polysemantic_neurons[:sample_size]
    
    if sampled_neurons:
        analyzer.analyze_linguistic_features(
            sampled_neurons,
            texts,
            k=20,  # Top 20 tokens per neuron
            batch_size=BATCH_SIZE
        )
    
    summary = analyzer.generate_summary()
    print(summary)
    
    analyzer.save_results(OUTPUT_DIR)

if __name__ == "__main__":
    main()
