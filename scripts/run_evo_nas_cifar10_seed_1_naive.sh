#!/bin/bash

# Training parameters for naive Evolutionary NAS
params= '{
  "results_path":"Results",
  "dataset": "cifar10",
  "parallel_processes": 5,
  "generations": 20,
  "population_size": 25,
  "nb_best_models_crossover": 5,
  "mutation_rate": 20, 
  "max_nb_feature_layers": 20,
  "max_nb_classification_layers": 2,
  "path_gene_pool": "gene_pool.txt",
  "path_rule_set": "rule_set.txt",
  "nb_epochs": 12,
  "gpu":true,
  "max_memory_footprint": 40000000,
  "seed": 1,
  "diversity_control": false,
  "soft_control": false
  }'

python main.py "$params"