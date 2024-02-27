import tensorflow as tf
import multiprocessing
import argparse
import warnings
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from utils.saver import Saver
from utils.loader import Loader
from utils import exporter
from nnalignment import alignment
import numpy as np
import time
from icecream import ic
import json

warnings.simplefilter('ignore')

def load_params(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params

params = load_params("params.json")

#################################
# define which dataset and experiment to use (--> adjust it in train.py and add more datasets there)
#################################DATASETS#####################################
#DATASETS = ["cifar10", "cifar100", "imagenet16-120"]

#EXPERIMENT = "paper_cifar10_s4_e1_d100_40mb_div"

#CLASSES_FILTER = []  # containing all classes that should be used for optimization
#SEED=4
np.random.seed(int(params["seed"]))
tf.random.set_seed(int(params["seed"]))
# select what dataset to use --> make sure the data loader is defined in datasets/get_datasets.py
#DATASET = DATASETS[2]

#################################
# define EvoNAS hyper-parameters
#################################
#NB_GENERATIONS = 20
#POPULATION_SIZE = 25
#NB_BEST_MODELS_CROSSOVER = 5  # specifies the number of models that will be used for crossover
#MUTATION_RATE = 20  # in percent

#MAX_NB_FEATURE_LAYERS = 20  # max number layers before GAP, GMP or Flatten layer in the first generation
#MAX_NB_CLASSIFICATION_LAYERS = 2  # max number layers after GAP, GMP or Flatten layer in the first generation

#PATH_GENE_POOL = "genetic_algorithm/src/models_structure/gene_pool.txt"
#PATH_RULE_SET = "genetic_algorithm/src/models_structure/rule_set.txt"


######################################
# define DNN training hyper-parameters
######################################
#NB_EPOCHS = 12
#MIN_FREE_SPACE_GPU = 5000000000  # 6 GB

#################################
# define constraints
#################################
#MAX_MEMORY_FOOTPRINT=40000000 #in Bytes (800000 bytes --> 40 MB)
#CONTROL_DIVERSITY=True
#SOFT_CONTROL=False

if params["dataset"]=="cifar10":    
    params["input_shape"] = (32,32,3)
    params["nb_classes"] = 10
    params["classes_filter"]=list(range(10))
elif params["dataset"]=="cifar100":
    params["input_shape"] = (32,32,3)
    params["nb_classes"] = 100
    params["classes_filter"]=list(range(100))
elif params["dataset"]=="imagenet16-120":
    params["input_shape"] = (16,16,3)
    params["nb_classes"] = 120
    params["classes_filter"]=list(range(120))


def main(continue_from=None):
    #tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

    my_saver= Saver(params["EXPERIMENT"],params["results_path"])

    if continue_from['continue_from_ga_run'] is None:
        my_ga = GeneticAlgorithm(params, my_saver)
        # random init the population of the first generation
        my_ga.init_first_generation()
        gen_start = 1

        # save params
        my_saver.save_params(params)
    else:
        ic("loader")
        my_loader = Loader(continue_from)
        params_ = my_loader.get_params()

        gen_start = my_loader.get_gen_start()

        my_ga = GeneticAlgorithm(params_, my_saver, my_loader)

        # save params
        my_saver.save_params(params_)
    elapsed_time={}

    for i_generation in range(gen_start, params["generations"]+1):
        start_time=time.time()

        my_ga.prepare_generation(i_generation)

        # Pre-selection of candidate chromosomes, which are trained on a GPU afterwards
        my_ga.evaluate_memory_footprint()

        # Create similarity matrix sim_total_generation and save alignments
        sim_total_generation, _,_,_,_=alignment.align_sequences(my_ga.population_genotype)
        last_median_similarity=np.median(sim_total_generation)
        alignment.align_sequences(chromosomes=my_ga.population_genotype, results_dir=str(my_ga.my_saver.results_dir), individuals_name=my_ga.individuals_names, generation=i_generation, from_folder=False)

        # train all neural networks
        my_ga.train_neural_networks()

        # Check that all the models were trained correctly, train again models that don´t have a reported fitness score.
        my_ga.check_accuracy_and_inference_time(i_generation)


        # determine the fitness for each model and select the best ones
        my_ga.selection()

        # Preparation of the next generation, unless we have just run the last generation
        if i_generation != params["generations"]:

            if params["diversity_control"]:
                my_ga.population_genotype, my_ga.parents_names=alignment.control_diversity(my_ga, last_median_similarity, i_generation, params["soft_control"])
            else:
                my_ga.crossover()
                my_ga.mutation()                

        end_time = time.time()
        elapsed_time[i_generation] = end_time - start_time

    my_saver.save_best_individuals_results_and_similarity(elapsed_time)
    exporter.merge_results(my_saver.results_dir, f'compressed_{params["EXPERIMENT"]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Evolutionary Neural Architecture Search',
        description='This package runs an evolutionary neural architecture search to find constrained DNN architectures.'
    )

    parser.add_argument('continue_from_ga_run', nargs='?', default=None)
    parser.add_argument('continue_from_generation', nargs='?', default=None)
    args = parser.parse_args()
    
    continue_from = {'continue_from_ga_run': args.continue_from_ga_run,
                     'continue_from_generation': args.continue_from_generation}
    main(continue_from)
