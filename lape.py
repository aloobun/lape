import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import logging
from typing import List, Dict, Tuple, Set, Any, Optional
import json
import os
from datetime import datetime
from collections import defaultdict, Counter
import string

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LAPE:
    def __init__(self, model_name: str, languages: List[str], device: str = "auto"):
        self.model_name = model_name
        self.languages = languages
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.n_layers = self.model.cfg.n_layers
        self.d_mlp = self.model.cfg.d_mlp
        self.total_neurons = self.n_layers * self.d_mlp
                
        self.activation_probabilities = {}
        self.lape_scores = None
        self.variance_scores = None
        self.neuron_types = {}
        self.language_specific_neurons = {}
        self.polysemantic_neurons = {}
        self.activation_magnitudes = {}
        self.top_activating_tokens = {}
        self.linguistic_features = {}
        
    def load_data(self, num_samples_per_lang: int = 1000, dataset_name: str = "soketlabs/bhasha-wiki"):
        data = {}
        for lang in self.languages:
            logging.info(f"Loading data for {lang} from {dataset_name}...")
            dataset = load_dataset(dataset_name, "20231101." + lang, split="train", streaming=True)
            texts = [item['text'] for item in dataset.take(num_samples_per_lang)]
            data[lang] = texts
        return data
    
    def get_activation_probabilities(
        self,
        texts: Dict[str, List[str]],
        batch_size: int = 8,
        activation_threshold: float = 0.01,
        use_percentile_threshold: bool = False,
        percentile_threshold: float = 95.0,
        use_magnitude: bool = False
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Computes activation probabilities for each neuron.
        """
        self.activation_probabilities = {}
        activation_thresholds = {}
        
        for lang in self.languages:
            logging.info(f"Analyzing language: {lang.upper()}")
            
            activation_counts = {i: np.zeros(self.d_mlp, dtype=np.float64) for i in range(self.n_layers)}
            magnitude_sums = {i: np.zeros(self.d_mlp, dtype=np.float64) for i in range(self.n_layers)}
            total_tokens_processed = 0
            
            act_name_filter = lambda name: "mlp.hook_post" in name
            
            # collect all activations to compute percentile threshold if needed
            all_activations = []
            if use_percentile_threshold:
                with torch.no_grad():
                    for i in tqdm(range(0, len(texts[lang]), batch_size), desc=f"Collecting {lang} activations"):
                        batch_texts = texts[lang][i:i+batch_size]
                        inputs = self.tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.model.cfg.n_ctx
                        ).to(self.device)
                        
                        _, cache = self.model.run_with_cache(inputs.input_ids, names_filter=act_name_filter)
                        attention_mask = inputs.attention_mask.bool()
                        
                        for layer_idx in range(self.n_layers):
                            activations = cache[f"blocks.{layer_idx}.mlp.hook_post"]
                            valid_activations = activations[attention_mask]
                            all_activations.append(valid_activations.cpu().numpy())
                
                # calculate percentile threshold
                all_activations = np.concatenate(all_activations)
                activation_threshold = np.percentile(all_activations, percentile_threshold)
            
            # compute activation probabilities with the threshold
            with torch.no_grad():
                for i in tqdm(range(0, len(texts[lang]), batch_size), desc=f"Processing {lang} batches"):
                    batch_texts = texts[lang][i:i+batch_size]
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.model.cfg.n_ctx
                    ).to(self.device)
                    
                    # Use run_with_cache to get internal activations
                    _, cache = self.model.run_with_cache(inputs.input_ids, names_filter=act_name_filter)
                    
                    attention_mask = inputs.attention_mask.bool()
                    
                    for layer_idx in range(self.n_layers):
                        # (batch, seq_len, d_mlp)
                        activations = cache[f"blocks.{layer_idx}.mlp.hook_post"]
                        
                        # non padding tokens
                        valid_activations = activations[attention_mask]  # (n_tokens, d_mlp)
                        
                        if use_magnitude:
                            # sum the magnitudes of activations
                            magnitude_sums[layer_idx] += valid_activations.sum(dim=0).cpu().numpy()
                        else:
                            positive_counts = (valid_activations > activation_threshold).sum(dim=0).cpu().numpy()
                            activation_counts[layer_idx] += positive_counts
                    
                    total_tokens_processed += attention_mask.sum().item()
            
            if total_tokens_processed == 0:
                self.activation_probabilities[lang] = {i: np.zeros(self.d_mlp) for i in range(self.n_layers)}
                continue
            
            # calculating probabilities by dividing counts by total tokens
            if use_magnitude:
                self.activation_probabilities[lang] = {
                    layer: sums / total_tokens_processed
                    for layer, sums in magnitude_sums.items()
                }
                self.activation_magnitudes[lang] = self.activation_probabilities[lang].copy()
            else:
                self.activation_probabilities[lang] = {
                    layer: counts / total_tokens_processed
                    for layer, counts in activation_counts.items()
                }
            
            activation_thresholds[lang] = activation_threshold
        
        self.activation_thresholds = activation_thresholds
        return self.activation_probabilities
    
    def calculate_lape_and_variance(self):     
        self.lape_scores = {i: np.zeros(self.d_mlp) for i in range(self.n_layers)}
        self.variance_scores = {i: np.zeros(self.d_mlp) for i in range(self.n_layers)}
        
        for layer_idx in tqdm(range(self.n_layers), desc="Calculating LAPE and variance"):
            for neuron_idx in range(self.d_mlp):
                prob_vector = np.array([
                    self.activation_probabilities[lang][layer_idx][neuron_idx] for lang in self.languages
                ])
                
                # variance
                self.variance_scores[layer_idx][neuron_idx] = np.var(prob_vector)
                
                # entropy
                prob_sum = prob_vector.sum()
                if prob_sum == 0:
                    self.lape_scores[layer_idx][neuron_idx] = 0.0
                    continue
                
                p_prime = prob_vector / prob_sum
                entropy = -np.sum(p_prime * np.log(p_prime + 1e-9))
                self.lape_scores[layer_idx][neuron_idx] = entropy
        
        return self.lape_scores, self.variance_scores
    
    def analyze_activation_distributions(self, num_bins: int = 100):
        if self.lape_scores is None:
            self.calculate_lape_and_variance()
        
        all_act_probs = []
        for lang in self.languages:
            for layer in range(self.n_layers):
                all_act_probs.extend(self.activation_probabilities[lang][layer])
        
        all_lape_scores = []
        for layer in range(self.n_layers):
            all_lape_scores.extend(self.lape_scores[layer])
        
        act_stats = {
            "min": np.min(all_act_probs),
            "max": np.max(all_act_probs),
            "mean": np.mean(all_act_probs),
            "median": np.median(all_act_probs),
            "std": np.std(all_act_probs),
            "percentiles": {
                "1%": np.percentile(all_act_probs, 1),
                "5%": np.percentile(all_act_probs, 5),
                "10%": np.percentile(all_act_probs, 10),
                "25%": np.percentile(all_act_probs, 25),
                "50%": np.percentile(all_act_probs, 50),
                "75%": np.percentile(all_act_probs, 75),
                "90%": np.percentile(all_act_probs, 90),
                "95%": np.percentile(all_act_probs, 95),
                "99%": np.percentile(all_act_probs, 99),
            }
        }
        
        lape_stats = {
            "min": np.min(all_lape_scores),
            "max": np.max(all_lape_scores),
            "mean": np.mean(all_lape_scores),
            "median": np.median(all_lape_scores),
            "std": np.std(all_lape_scores),
            "percentiles": {
                "1%": np.percentile(all_lape_scores, 1),
                "5%": np.percentile(all_lape_scores, 5),
                "10%": np.percentile(all_lape_scores, 10),
                "25%": np.percentile(all_lape_scores, 25),
                "50%": np.percentile(all_lape_scores, 50),
                "75%": np.percentile(all_lape_scores, 75),
                "90%": np.percentile(all_lape_scores, 90),
                "95%": np.percentile(all_lape_scores, 95),
                "99%": np.percentile(all_lape_scores, 99),
            }
        }
        
        act_hist, act_bins = np.histogram(all_act_probs, bins=num_bins)
        lape_hist, lape_bins = np.histogram(all_lape_scores, bins=num_bins)
        
        return {
            "activation_probabilities": {
                "statistics": act_stats,
                "histogram": {
                    "counts": act_hist.tolist(),
                    "bins": act_bins.tolist()
                }
            },
            "lape_scores": {
                "statistics": lape_stats,
                "histogram": {
                    "counts": lape_hist.tolist(),
                    "bins": lape_bins.tolist()
                }
            }
        }
    
    def classify_neurons(
        self,
        lape_percentile_low: float = 1.0,
        lape_percentile_high: float = 99.0,
        activation_prob_threshold: float = 0.01,
        agnostic_threshold: float = 0.8,
        polysemantic_threshold: float = 0.3
    ):
        if self.lape_scores is None:
            self.calculate_lape_and_variance()
        
        all_scores_flat = np.concatenate([scores for scores in self.lape_scores.values()])
        lape_threshold_low = np.percentile(all_scores_flat, lape_percentile_low)
        lape_threshold_high = np.percentile(all_scores_flat, lape_percentile_high)
        
        self.neuron_types = {
            "language_specific": [],
            "language_agnostic": [],
            "multi_language": [],
            "inactive": [],
            "polysemantic": []
        }
        
        for layer_idx in tqdm(range(self.n_layers), desc="Classifying neurons"):
            for neuron_idx in range(self.d_mlp):
                # activation probabilities for this neuron across all languages
                probs = np.array([self.activation_probabilities[lang][layer_idx][neuron_idx] for lang in self.languages])
                
                # checking how many languages this neuron is active in
                active_langs = np.sum(probs > activation_prob_threshold)
                fraction_active = active_langs / len(self.languages)
                
                # get the LAPE score
                lape_score = self.lape_scores[layer_idx][neuron_idx]
                
                if fraction_active == 0:
                    self.neuron_types["inactive"].append((layer_idx, neuron_idx))
                elif lape_score <= lape_threshold_low and active_langs >= 1:
                    # low entropy and active in at least one language
                    self.neuron_types["language_specific"].append((layer_idx, neuron_idx))
                elif lape_score >= lape_threshold_high and fraction_active >= agnostic_threshold:
                    # high entropy and active in most languages
                    self.neuron_types["language_agnostic"].append((layer_idx, neuron_idx))
                elif active_langs > 1 and fraction_active < agnostic_threshold:
                    # active in multiple but not all languages
                    self.neuron_types["multi_language"].append((layer_idx, neuron_idx))
                    
                    if active_langs >= polysemantic_threshold * len(self.languages):
                        active_probs = probs[probs > activation_prob_threshold]
                        if len(active_probs) > 1:
                            cv = np.std(active_probs) / (np.mean(active_probs) + 1e-9)
                            if cv < 0.5:  # Arbitrary threshold for "similar" activation
                                self.neuron_types["polysemantic"].append((layer_idx, neuron_idx))
                else:
                    self.neuron_types["inactive"].append((layer_idx, neuron_idx))
        
        self.identify_language_specific_neurons(activation_prob_threshold)
        
        self.classification_thresholds = {
            "lape_percentile_low": lape_percentile_low,
            "lape_percentile_high": lape_percentile_high,
            "activation_prob_threshold": activation_prob_threshold,
            "agnostic_threshold": agnostic_threshold,
            "polysemantic_threshold": polysemantic_threshold,
            "lape_threshold_low": lape_threshold_low,
            "lape_threshold_high": lape_threshold_high
        }
        
        return self.neuron_types
    
    def identify_language_specific_neurons(self, activation_prob_threshold: float = 0.01):
        self.language_specific_neurons = {lang: [] for lang in self.languages}
        
        for layer_idx, neuron_idx in self.neuron_types["language_specific"]:
            # check high activation prob
            max_prob = 0
            max_lang = None
            
            for lang in self.languages:
                prob = self.activation_probabilities[lang][layer_idx][neuron_idx]
                if prob > max_prob and prob > activation_prob_threshold:
                    max_prob = prob  # Store the probability value, not the language
                    max_lang = lang  # Store the language separately
            
            if max_lang:
                self.language_specific_neurons[max_lang].append((layer_idx, neuron_idx))
        
        return self.language_specific_neurons
    
    def analyze_polysemantic_neurons(self, activation_prob_threshold: float = 0.01):
        polysemantic_analysis = {}
        
        for layer_idx, neuron_idx in self.neuron_types["polysemantic"]:
            probs = {lang: self.activation_probabilities[lang][layer_idx][neuron_idx] for lang in self.languages}
            
            sorted_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            active_langs = [lang for lang, prob in sorted_langs if prob > activation_prob_threshold]
            active_probs = [prob for lang, prob in sorted_langs if prob > activation_prob_threshold]
            
            if len(active_langs) > 1:
                cv = np.std(active_probs) / (np.mean(active_probs) + 1e-9)
                p_prime = np.array(active_probs) / sum(active_probs)
                entropy = -np.sum(p_prime * np.log(p_prime + 1e-9))
                
                polysemantic_analysis[(layer_idx, neuron_idx)] = {
                    "active_languages": active_langs,
                    "activation_probabilities": {lang: prob for lang, prob in sorted_langs if prob > activation_prob_threshold},
                    "coefficient_of_variation": cv,
                    "entropy": entropy,
                    "num_active_languages": len(active_langs)
                }
        
        self.polysemantic_neurons = polysemantic_analysis
        return polysemantic_analysis
    
    def analyze_layer_distribution(self):      
        layer_distribution = {
            neuron_type: [0] * self.n_layers
            for neuron_type in self.neuron_types
        }
        
        for neuron_type, neurons in self.neuron_types.items():
            for layer_idx, _ in neurons:
                layer_distribution[neuron_type][layer_idx] += 1
        
        # Calculate percentages
        layer_percentages = {}
        for neuron_type, counts in layer_distribution.items():
            layer_percentages[neuron_type] = [
                (count / self.d_mlp) * 100 for count in counts
            ]
        
        return {
            "counts": layer_distribution,
            "percentages": layer_percentages
        }
    
    def get_top_activating_tokens(
        self,
        neuron_set: List[Tuple[int, int]],
        texts: Dict[str, List[str]],
        k: int = 20,
        batch_size: int = 8,
        activation_threshold: float = 0.0
    ) -> Dict[Tuple[int, int], List[Tuple[str, float]]]:
        if not neuron_set:
            return {}
        
        neuron_activations = {(layer, neuron): [] for layer, neuron in neuron_set}
        
        act_name_filter = lambda name: "mlp.hook_post" in name
        
        for lang in self.languages:
            logging.info(f"Processing {lang} for top activating tokens")
            
            with torch.no_grad():
                for i in tqdm(range(0, len(texts[lang]), batch_size), desc=f"Processing {lang} batches"):
                    batch_texts = texts[lang][i:i+batch_size]
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.model.cfg.n_ctx,
                        return_offsets_mapping=True
                    ).to(self.device)
                    
                    input_ids = inputs.input_ids
                    attention_mask = inputs.attention_mask.bool()
                    
                    _, cache = self.model.run_with_cache(input_ids, names_filter=act_name_filter)
                    
                    for layer_idx, neuron_idx in neuron_set:
                        activations = cache[f"blocks.{layer_idx}.mlp.hook_post"]
                        
                        neuron_act = activations[:, :, neuron_idx]
                        
                        valid_act = neuron_act[attention_mask]
                        valid_ids = input_ids[attention_mask]
                        
                        valid_act_np = valid_act.cpu().numpy()
                        valid_ids_np = valid_ids.cpu().numpy()
                        
                        mask = valid_act_np > activation_threshold
                        filtered_act = valid_act_np[mask]
                        filtered_ids = valid_ids_np[mask]
                        
                        for token_id, act_val in zip(filtered_ids, filtered_act):
                            token = self.tokenizer.decode([token_id])
                            neuron_activations[(layer_idx, neuron_idx)].append((token, act_val, lang))
        
        top_tokens = {}
        for neuron, activations in neuron_activations.items():
            if not activations:
                top_tokens[neuron] = []
                continue
            
            token_activations = defaultdict(list)
            for token, act_val, lang in activations:
                token_activations[token].append((act_val, lang))
            
            token_mean_act = {}
            token_languages = {}
            for token, act_lang_pairs in token_activations.items():
                act_vals = [pair[0] for pair in act_lang_pairs]
                langs = [pair[1] for pair in act_lang_pairs]
                token_mean_act[token] = np.mean(act_vals)
                token_languages[token] = list(set(langs))  # Unique languages
            
            sorted_tokens = sorted(token_mean_act.items(), key=lambda x: x[1], reverse=True)[:k]
            
            top_tokens[neuron] = [(token, act_val, token_languages[token]) for token, act_val in sorted_tokens]
        
        self.top_activating_tokens = top_tokens
        return top_tokens
    
    def analyze_linguistic_features(
        self,
        neuron_set: List[Tuple[int, int]],
        texts: Dict[str, List[str]],
        k: int = 20,
        batch_size: int = 8
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        top_tokens = self.get_top_activating_tokens(neuron_set, texts, k, batch_size)
        
        linguistic_features = {}
        
        for neuron, tokens in top_tokens.items():
            if not tokens:
                linguistic_features[neuron] = {"error": "No activating tokens found"}
                continue
            
            feature_analysis = {
                "top_tokens": tokens,
                "token_types": {
                    "punctuation": 0,
                    "digits": 0,
                    "letters": 0,
                    "whitespace": 0,
                    "other": 0
                },
                "token_lengths": [],
                "unique_tokens": len(tokens),
                "languages": defaultdict(int)
            }
            
            for token, act_val, langs in tokens:
                for lang in langs:
                    feature_analysis["languages"][lang] += 1
                
                if all(c in string.punctuation for c in token):
                    feature_analysis["token_types"]["punctuation"] += 1
                elif all(c in string.digits for c in token):
                    feature_analysis["token_types"]["digits"] += 1
                elif all(c in string.whitespace for c in token):
                    feature_analysis["token_types"]["whitespace"] += 1
                elif all(c.isalpha() or c in ["'", "-", " "] for c in token):
                    feature_analysis["token_types"]["letters"] += 1
                else:
                    feature_analysis["token_types"]["other"] += 1
                
                feature_analysis["token_lengths"].append(len(token))
            
            if feature_analysis["token_lengths"]:
                feature_analysis["avg_token_length"] = np.mean(feature_analysis["token_lengths"])
                feature_analysis["max_token_length"] = np.max(feature_analysis["token_lengths"])
                feature_analysis["min_token_length"] = np.min(feature_analysis["token_lengths"])
            else:
                feature_analysis["avg_token_length"] = 0
                feature_analysis["max_token_length"] = 0
                feature_analysis["min_token_length"] = 0
            
            linguistic_features[neuron] = feature_analysis
        
        self.linguistic_features = linguistic_features
        return linguistic_features
    
    def save_results(self, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for lang in self.languages:
            np.save(
                os.path.join(output_dir, f"{timestamp}_activation_probabilities_{lang}.npy"),
                np.array([self.activation_probabilities[lang][layer] for layer in range(self.n_layers)])
            )
        
        if self.lape_scores is not None:
            np.save(
                os.path.join(output_dir, f"{timestamp}_lape_scores.npy"),
                np.array([self.lape_scores[layer] for layer in range(self.n_layers)])
            )
        
        if self.variance_scores is not None:
            np.save(
                os.path.join(output_dir, f"{timestamp}_variance_scores.npy"),
                np.array([self.variance_scores[layer] for layer in range(self.n_layers)])
            )
        
        if self.neuron_types:
            with open(os.path.join(output_dir, f"{timestamp}_neuron_types.json"), "w") as f:
                json_neuron_types = {
                    neuron_type: [f"{layer},{neuron}" for layer, neuron in neurons]
                    for neuron_type, neurons in self.neuron_types.items()
                }
                json.dump(json_neuron_types, f, indent=2)
        
        if self.language_specific_neurons:
            with open(os.path.join(output_dir, f"{timestamp}_language_specific_neurons.json"), "w") as f:
                json_lang_specific = {
                    lang: [f"{layer},{neuron}" for layer, neuron in neurons]
                    for lang, neurons in self.language_specific_neurons.items()
                }
                json.dump(json_lang_specific, f, indent=2)
        
        if self.polysemantic_neurons:
            with open(os.path.join(output_dir, f"{timestamp}_polysemantic_neurons.json"), "w") as f:
                json_polysemantic = {
                    f"{layer},{neuron}": data
                    for (layer, neuron), data in self.polysemantic_neurons.items()
                }
                json.dump(json_polysemantic, f, indent=2)
        
        if self.top_activating_tokens:
            with open(os.path.join(output_dir, f"{timestamp}_top_activating_tokens.json"), "w") as f:
                json_top_tokens = {}
                for (layer, neuron), tokens in self.top_activating_tokens.items():
                    serializable_tokens = []
                    for token, act_val, langs in tokens:
                        serializable_tokens.append({
                            "token": token,
                            "activation": float(act_val),
                            "languages": langs
                        })
                    json_top_tokens[f"{layer},{neuron}"] = serializable_tokens
                json.dump(json_top_tokens, f, indent=2)
        
        if self.linguistic_features:
            with open(os.path.join(output_dir, f"{timestamp}_linguistic_features.json"), "w") as f:
                json_ling_features = {}
                for (layer, neuron), features in self.linguistic_features.items():
                    json_features = features.copy()
                    if "languages" in json_features and isinstance(json_features["languages"], defaultdict):
                        json_features["languages"] = dict(json_features["languages"])
                    
                    if "top_tokens" in json_features:
                        serializable_top_tokens = []
                        for token, act_val, langs in json_features["top_tokens"]:
                            serializable_top_tokens.append({
                                "token": token,
                                "activation": float(act_val),
                                "languages": langs
                            })
                        json_features["top_tokens"] = serializable_top_tokens
                    
                    json_ling_features[f"{layer},{neuron}"] = json_features
                json.dump(json_ling_features, f, indent=2)
        
        layer_dist = self.analyze_layer_distribution()
        if layer_dist:
            with open(os.path.join(output_dir, f"{timestamp}_layer_distribution.json"), "w") as f:
                json.dump(layer_dist, f, indent=2)
        
        thresholds = {
            "activation_thresholds": getattr(self, "activation_thresholds", {}),
            "classification_thresholds": getattr(self, "classification_thresholds", {})
        }
        with open(os.path.join(output_dir, f"{timestamp}_thresholds.json"), "w") as f:
            json.dump(thresholds, f, indent=2)
        
        summary = self.generate_summary()
        with open(os.path.join(output_dir, f"{timestamp}_summary.txt"), "w") as f:
            f.write(summary)
            
    def generate_summary(self) -> str:
        summary = []
        summary.append("="*50)
        summary.append("Neuron Analysis Summary")
        summary.append("="*50)
        summary.append(f"Model: {self.model_name}")
        summary.append(f"Languages Analyzed: {', '.join(self.languages)}")
        
        if hasattr(self, "classification_thresholds"):
            summary.append("Classification Criteria:")
            summary.append(f"  - Language-specific: LAPE in bottom {self.classification_thresholds['lape_percentile_low']:.1f}% and active in at least one language")
            summary.append(f"  - Language-agnostic: LAPE in top {self.classification_thresholds['lape_percentile_high']:.1f}% and active in at least {self.classification_thresholds['agnostic_threshold']*100:.0f}% of languages")
            summary.append(f"  - Multi-language: Active in multiple but not all languages")
            summary.append(f"  - Polysemantic: Active in multiple languages with similar activation patterns")
            summary.append(f"  - Inactive: Activation probability < {self.classification_thresholds['activation_prob_threshold']:.3f} for all languages")
        
        summary.append("-"*50)
        summary.append(f"Total MLP neurons in model: {self.total_neurons:,}")
        
        if self.neuron_types:
            summary.append("\nNumber of neurons by type:")
            for neuron_type, neurons in self.neuron_types.items():
                count = len(neurons)
                percentage = (count / self.total_neurons) * 100
                summary.append(f"  - {neuron_type.replace('_', ' ').title()}: {count:,} neurons ({percentage:.2f}%)")
        
        if self.language_specific_neurons:
            summary.append("\nNumber of specific neurons identified per language:")
            for lang, neurons in self.language_specific_neurons.items():
                count = len(neurons)
                percentage = (count / self.total_neurons) * 100
                summary.append(f"  - {lang.upper()}: {count:,} neurons ({percentage:.2f}%)")
            
            if len(self.languages) >= 2:
                summary.append("\nOverlap between language-specific neurons:")
                for i, lang1 in enumerate(self.languages):
                    for lang2 in self.languages[i+1:]:
                        set1 = set(self.language_specific_neurons[lang1])
                        set2 = set(self.language_specific_neurons[lang2])
                        overlap = len(set1.intersection(set2))
                        if len(set1) > 0 and len(set2) > 0:
                            overlap_pct = (overlap / min(len(set1), len(set2))) * 100
                            summary.append(f"  - {lang1.upper()} & {lang2.upper()}: {overlap} neurons ({overlap_pct:.1f}% of smaller set)")
        
        if self.polysemantic_neurons:
            summary.append(f"\nNumber of polysemantic neurons: {len(self.polysemantic_neurons):,}")
            
            lang_counts = {lang: 0 for lang in self.languages}
            for neuron_data in self.polysemantic_neurons.values():
                for lang in neuron_data["active_languages"]:
                    lang_counts[lang] += 1
            
            summary.append("Languages most represented in polysemantic neurons:")
            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                summary.append(f"  - {lang.upper()}: {count} neurons")
        
        layer_dist = self.analyze_layer_distribution()
        if layer_dist and layer_dist.get("percentages"):
            summary.append("\nLayer distribution of language-specific neurons (% of layer neurons):")
            lang_specific_pct = layer_dist["percentages"].get("language_specific", [0] * self.n_layers)
            for layer_idx, pct in enumerate(lang_specific_pct):
                summary.append(f"  - Layer {layer_idx}: {pct:.2f}%")
        
        if self.linguistic_features:
            summary.append("\nLinguistic Features Analysis (for polysemantic neurons):")
            
            token_types = {
                "punctuation": 0,
                "digits": 0,
                "letters": 0,
                "whitespace": 0,
                "other": 0
            }
            
            avg_lengths = []
            
            for features in self.linguistic_features.values():
                if "token_types" in features:
                    for ttype, count in features["token_types"].items():
                        token_types[ttype] += count
                
                if "avg_token_length" in features:
                    avg_lengths.append(features["avg_token_length"])
            
            total_tokens = sum(token_types.values())
            if total_tokens > 0:
                summary.append("Token type distribution:")
                for ttype, count in token_types.items():
                    pct = (count / total_tokens) * 100
                    summary.append(f"  - {ttype}: {pct:.1f}%")
            
            if avg_lengths:
                summary.append(f"Average token length: {np.mean(avg_lengths):.2f}")
        
        summary.append("="*50)
        return "\n".join(summary)
