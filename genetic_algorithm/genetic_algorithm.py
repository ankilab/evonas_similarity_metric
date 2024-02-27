import os.path
import signal
from subprocess import Popen
import math
import pandas as pd
from coolname import generate_slug
import json
import numpy as np
import time
import subprocess
import nvidia_smi
from multiprocessing import Pool
from icecream import ic
import warnings
warnings.simplefilter('ignore')
#############
from datasets.get_datasets import get_datasets
import tensorflow as tf
import os
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import json
import time
import warnings
warnings.simplefilter('ignore')

##############

#from .src.genepool import GenePool
######################
from .src.models_structure.genepool import GenePool
from .src.models_structure.translation import translate
######################
#from .src.translation import translate
#from ..utils import saver
#from ..utils import loader
#from utils.saver import Saver
#from utils.loader import Loader
from .src.fitness import calculate_fitness



class GeneticAlgorithm:
    def __init__(self, params: dict, saver=None, loader= None):
        self.params = params
        self.my_saver = saver
        self.my_gene_pool = GenePool(params)

        # define all variables that change after each generation
        if loader is None:
            self.population_genotype = None  # dicts containing all individuals with its properties
        else:
            # load all genotypes --> each of them will get a new name but that's the easiest solution I found for now
            self.population_genotype = loader.load_population_genotype()

        self.individuals_names = None  # randomly created names for all individuals within one generation
        self.preselected_individuals = None  # preselected individuals (names) after accuracy/memory footprint determination
        self.parents_names = None  # list containing the chromosome names of the parents that yielded to a new chromosome
        self.population_phenotype = None  # generated TF models
        self.population_phenotype_tflite = None  # generated TFLite models
        self.generation_counter = None  # information in which generation we are currently
        self.population_next_generation = None  # contains all individuals for next generation (determined through selection, crossover and mutation)
        self.best_models_current_generation = None  # contains the best models of the current generation

    def init_first_generation(self):
        self.population_genotype = []
        for _ in range(self.params['population_size']):
            random_chromosome = self.my_gene_pool.get_random_chromosome()
            self.population_genotype.append(random_chromosome)
        ic(len(self.population_genotype))

    def evaluate_memory_footprint(self):
        """ Load memory footprint after converting the model to TFLite.
        Determine models (dependent on the thresholds specified in main.py) that will be further evaluated. """
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'
        size_list=[]
        for individual in self.individuals_names:
            # read h5 model memory footprint
            model_path = path + individual + '/models/model_untrained.h5'
            memory_footprint_h5 = os.path.getsize(model_path)
            print(memory_footprint_h5)
            d = {'memory_footprint_h5': memory_footprint_h5}

            # take only the models into account that are below a certain threshold
            if memory_footprint_h5 <= self.params['max_memory_footprint']:
                self.preselected_individuals.append(individual)
                size_list.append(memory_footprint_h5)
            else:
                # set fitness directly to zero since it is not relevant anymore
                d["fitness"] = 0
            #self.preselected_individuals.append(individual)

            with open(path + individual + '/results.json', 'w') as f:
                json.dump(d, f, indent=2)


        combined_lists = list(zip(self.preselected_individuals, size_list))
        sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

        json_filename = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/sorted_combined_lists.json'
        with open(json_filename, 'w') as json_file:
            json.dump(sorted_combined_lists, json_file)

        sorted_list1, sorted_list2 = map(list,zip(*sorted_combined_lists))
        new_preselected_list = []

        while sorted_list1:
            # Take the last and first elements
            last_element = sorted_list1.pop()
            print("l",last_element)
            if not sorted_list1:
                new_preselected_list.extend([last_element])
                break
            first_element = sorted_list1.pop(0)
            print("f",first_element)

            # Append them to the new list
            new_preselected_list.extend([first_element, last_element])

        self.preselected_individuals=new_preselected_list
        if len(self.preselected_individuals) == 0:
            raise Exception("All models are too big in terms of file size. Therefore none of the generated models will"
                            " be further evaluated. Think about adjusting your GA parameters.")

    def train_neural_networks(self):
        with open(f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/preselected.txt', 'w') as file:
            # Write each element of the list to a new line
            for item in self.preselected_individuals:
                file.write("%s\n" % item)
        # train all neural networks
        # --> start several training processes here
        min_free_space = self.params['min_free_space_gpu']
        procs = []
        idx = 0

        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        n_training=math.ceil(len(self.preselected_individuals)/5)
        while idx < len(self.preselected_individuals):
            procs = [p for p in procs if p.poll() is None]
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            ic(idx)
            ic(info.free)
            if info.free > min_free_space and len(procs)<=5:
                command = 'python genetic_algorithm/src/train.py ' + \
                            f'--results_dir {self.my_saver.results_dir} ' + \
                            f'--gen_dir Generation_{self.generation_counter} ' + \
                            f'--individual_dir ' + ' '.join(self.preselected_individuals[idx:idx+n_training])+' '+ \
                            f'--nb_epochs {self.params["nb_epochs"]} ' + \
                            f'--dataset {self.params["dataset"]} ' + \
                            f'--classes_filter ' + ' '.join(str(i) for i in self.params["classes_filter"])
                procs.append(Popen(command, shell=True))
                idx += n_training
            #else:
                #for p in procs[:2]:
                #    p.wait()
            #    print(info.free)
            time.sleep(15)

        
        # make sure to wait until all processes are finished
        for p in procs:
            try:
                return_code = p.wait()
                if return_code != 0:
                    print(f"Warning: Process returned non-zero exit code: {return_code}")
            except Exception as e:
                print(f"Error waiting for process: {e}")

        nvidia_smi.nvmlShutdown()
    def selection(self):
        # calculate fitness of all preselected models
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'
        models_with_fitness = dict()
        for individual in self.preselected_individuals:
            with open(path + individual + '/results.json', 'r') as f:
                results = json.loads(f.read())
                fitness = calculate_fitness(results, self.params)

                # save fitness together with the individual name to sort and select them later
                models_with_fitness[f'{individual}'] = fitness

            # save fitness in results.json
            with open(path + individual + '/results.json', 'w') as f:
                results['fitness'] = float(fitness)
                json.dump(results, f, indent=2)

        # sort them by their achieved fitness
        models_with_fitness = sorted(models_with_fitness.items(), key=lambda item: item[1], reverse=True)

        # save the model with the best fitness to find it easier later on
        self.my_saver.save_best_individual(self.generation_counter, models_with_fitness[0])

        self.best_models_current_generation = \
            [models_with_fitness[i][0] for i in range(self.params['nb_best_models_crossover']) if
             i < len(models_with_fitness)]

    def crossover(self, n_populations=1):
        """ Crossover the best chromosomes to get the population for the next generation. """
        self.population_next_generation, self.parents_names = self.my_gene_pool.crossover(
            path=f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/',
            fittest_chromosomes=self.best_models_current_generation,
            n_populations=n_populations)

    def mutation(self, rate=None):
        """ Mutation of the population previously generated by crossover. """
        self.population_genotype = []
        for chromosome in self.population_next_generation:
            mutated_chromosome = self.my_gene_pool.mutate_chromosome(chromosome, rate=rate)
            self.population_genotype.append(mutated_chromosome)


    def _process_model_conversion(self, model):
        pass

    def prepare_generation(self, current_generation: int):
        self.individuals_names = []
        self.population_phenotype = []
        self.population_phenotype_tflite = []
        self.preselected_individuals = []
        self.best_models_current_generation = []
        self.population_next_generation = []

        self._generate_individuals_names()
        if self.parents_names is not None:
            self.my_saver.save_parents(self.generation_counter, self.individuals_names, self.parents_names)
            self.parents_names = None

        self.generation_counter = current_generation

        #d = {}
        #pool = Pool(os.cpu_count())
        #pool.map(self._process_model_conversion, )

        # translate all chromosomes: genotype (chromosome) --> phenotype (tf.keras.Model)
        params=self.params
        for chromosome in self.population_genotype:
            try:
                model = translate(chromosome, params)
                               
            except:
                raise ValueError(f"Error when translating from genotype to phenotype. Chromosome: {chromosome}")

            self.population_phenotype.append(model)

        # save untrained networks such that it can be loaded in a new process
        self.my_saver.save_chromosomes(self.population_genotype, self.population_phenotype,
                                       self.individuals_names, self.generation_counter)

    def _generate_individuals_names(self):
        self.individuals_names = []
        for _ in range(self.params['population_size']):
            # Assigning a random name to an individual to make it easier to track results
            random_name = generate_slug(2).replace("-", "_")
            while random_name in self.individuals_names:  # --> make sure that the random name does not already exist
                random_name = generate_slug(2).replace("-", "_")
            self.individuals_names.append(random_name)
    def check_accuracy_and_inference_time(self, generation):
        #root_folder="/data/pa94xuro/sequence_alignment/Results/ga_20231014-104814_random_training_speech_12_2D_wo_sim"
        #preselected_individuals=[f.name for f in os.scandir(f'{root_folder}/Generation_{str(generation)}') if f.is_dir()]
        def load_tf_model(path):
            m = tf.keras.models.load_model(path, custom_objects={"InstanceNormalization": InstanceNormalization,
                                                              }, compile=False)
            return m
        ds_train, ds_val, ds_test = get_datasets(self.params["dataset"], classes_filter=self.params["classes_filter"] )

        root_folder=self.my_saver.results_dir
        untrained_models=[]
        for name in self.preselected_individuals:
            with open(f'{root_folder}/Generation_{str(generation)}/{name}/results.json') as f:
                d = json.loads(f.read())
            
            if ("val_acc" in d.keys()) and ("inference_time" in d.keys()):
                ic("---Correctly trained models-------")
                ic(name)
                pass 
            else:
                ic("---Uncorrectly trained models-------")
                ic(name)
                untrained_models.append(name)

        procs = []
        idx = 0
        print("###### Start training untrained models #####")
        while idx < len(untrained_models):
            procs = [p for p in procs if p.poll() is None]
            ic("untrained model: ",untrained_models[idx])
            if len(procs)<=5:
                command = 'python genetic_algorithm/src/train.py ' + \
                            f'--results_dir {self.my_saver.results_dir} ' + \
                            f'--gen_dir Generation_{self.generation_counter} ' + \
                            f'--individual_dir ' + ' '.join(untrained_models[idx:idx+int(len(untrained_models)/5)+1])+' '+ \
                            f'--nb_epochs {self.params["nb_epochs"]} ' + \
                            f'--dataset {self.params["dataset"]} ' + \
                            f'--classes_filter ' + ' '.join(str(i) for i in self.params["classes_filter"])
                procs.append(Popen(command, shell=True))
                idx += int(len(untrained_models)/5)+1
            #else:
                #for p in procs[:2]:
                #    p.wait()
            #    print(info.free)
            time.sleep(15)

        
        # make sure to wait until all processes are finished
        for p in procs:
            try:
                return_code = p.wait()
                if return_code != 0:
                    print(f"Warning: Process returned non-zero exit code: {return_code}")
            except Exception as e:
                print(f"Error waiting for process: {e}")
