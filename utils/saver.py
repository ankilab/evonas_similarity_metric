import os
import time
import json
import csv
import shutil
from pathlib import Path
import pandas as pd
import sys
sys.path.append('../genetic_algorithm')
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from nnalignment import alignment


class Saver:
    def __init__(self, experiment, results_path):
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        self.results_dir = Path(f"{results_path}/ga_{time.strftime('%Y%m%d-%H%M%S')}_{experiment}")
        os.mkdir(self.results_dir)

        self.random_names = []

    def save_params(self, params):
        with open(self.results_dir / 'params.json', 'w') as f:
            json.dump(params, f, indent=4)

        # save genepool.txt and rule_set.txt
        shutil.copyfile(params["path_gene_pool"], self.results_dir / "gene_pool.txt")
        shutil.copyfile(params["path_rule_set"], self.results_dir / "rule_set.txt")

    def _get_path(self, gen_count, name=None):
        if name is None:
            return self.results_dir / f'Generation_{gen_count}'
        else:
            return self.results_dir / f'Generation_{gen_count}/{name}'

    def save_chromosomes(self, population_genotype: list, population_phenotype: list,
                         chromosome_names: list, gen_count: int) -> None:
        # create generation dir
        os.mkdir(self._get_path(gen_count))
        # generate individual dir and safe chromosome
        for name, chromosome_g, chromosome_p \
                in zip(chromosome_names, population_genotype, population_phenotype):

            p = self._get_path(gen_count, name)
            os.mkdir(p)
            self._save_chromosome_genotype(chromosome_g, p)
            p = p / "models"
            os.mkdir(p)
            self._save_chromosome_phenotype(chromosome_p, p)

            # convert tflite model to C-array
            #subprocess.call("xxd -i " + str(p / "model_tflite_untrained.tflite") + " > " + str(p / "model_c_array_untrained.cc"), shell=True)

    @staticmethod
    def _save_chromosome_genotype(chromosome, path):
        with open(path / 'chromosome.json', 'w') as f:
            json.dump(chromosome, f, indent=2)

    @staticmethod
    def _save_chromosome_phenotype(model_untrained, path):
        model_untrained.save(path / "model_untrained.h5")


    def save_best_individual(self, gen_count, best_individual):
        row = [f'Generation_{gen_count}', best_individual[0], best_individual[1]]
        with open(self.results_dir / r'best_individual_each_generation.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def save_parents(self, gen_count, individuals_new_generation, parents_names):
        for individual, (parent_1, parent_2, parent_1_split, parent_2_split) in zip(individuals_new_generation, parents_names):
            # parent_1_split and parent_2_split are the idx where the chromosomes are split up
            row = [f'Generation: {gen_count}', f'Parent_1: ({parent_1}, {parent_1_split})',
                   f'Parent_2: ({parent_2}, {parent_2_split})', f'New_Individual: {individual}']
            with open(self.results_dir / r'crossover_parents.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    def save_best_individuals_results_and_similarity(self, elapsed_time, results_dir=None):
            if results_dir is not None:
                save_folder=results_dir 
            else:
                save_folder=self.results_dir

            best_individuals=pd.read_csv(f'{save_folder}/best_individual_each_generation.csv', header=None, names=["Generation", "Name", "fitness"])
            mem=[]
            val=[]
            infer=[]
            chrom=[]
            for name in best_individuals.Name:
                results_path=f'{save_folder}/{best_individuals[best_individuals.Name==name]["Generation"].values[0]}/{name}'
                with open(f'{results_path}/results.json', 'r') as json_file:
                    dict1= json.load(json_file)
                mem.append(dict1["memory_footprint_h5"])
                val.append(dict1["val_acc"])
                try:
                    infer.append(dict1["inference_time"])
                except:
                    print(name)
                    infer.append(None)
                
                with open(f'{results_path}/chromosome.json', 'r') as json_file:
                    chrom.append(json.load(json_file))
            best_individuals["val_acc"]=val
            best_individuals["memory_footprint_h5"]=mem
            best_individuals["inference_time"]=infer
            #best_individuals["fitness"]=best_individuals["val_acc"]
            best_individuals['Generation'] = best_individuals['Generation'].str[11:]
            best_individuals["Generation"] = best_individuals["Generation"].astype(int)
            if len(list(elapsed_time.keys()))==len(best_individuals):
                best_individuals['elapsed time(s)'] = best_individuals['Generation'].map(elapsed_time)
                best_individuals['elapsed time(h)'] = best_individuals['elapsed time(s)'].apply(self.seconds_to_hms_string)
            best_individuals.to_csv(f'{save_folder}/best_individuals.csv', index=False)
            ########################## Best individuals alignment #####
            chrom=[]
            for name in best_individuals.Name:
                results_path=f'{save_folder}/Generation_{best_individuals[best_individuals.Name==name]["Generation"].values[0]}/{name}'
                with open(f'{results_path}/chromosome.json', 'r') as json_file:
                    chrom.append(json.load(json_file))
            alignment.align_sequences(chrom, individuals_name=best_individuals.Name.tolist(), results_dir=save_folder)
            ########################### Similarity scores ##############
            #best_sim_df, _, _, _, _=alignment.align_sequences(chrom, individuals_name=best_individuals.Name.tolist())
            #print(best_sim_df.head())
            #best_sim_df.to_csv(f'{self.results_dir}/similarity_best_individuals.csv', index=True)
    def seconds_to_hms_string(self,seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


if __name__ == '__main__':
#    folders=["ga_20240201-174818_paper_cifar100_p25_s7_e2_d100_40mb_div", "ga_20240202-081945_paper_imagenet16_p25_s1_e2_d100_40mb_div","ga_20240202-082246_paper_imagenet16_p25_s1_e20_d100_40mb_div","ga_20240203-085039_paper_imagenet16_p25_s2_e20_d100_40mb_div","ga_20240204-092401_paper_cifar100_p25_s2_e2_d100_40mb_div"]
    

    #folders=["ga_20240131-080749_paper_imagenet16_p50_s5_e20_d100"]
    folders=["ga_20240221-142800_paper_cifar10_s4_e1_d100_40mb_div_soft","ga_20240225-014806_paper_cifar10_s4_e1_d100_40mb_div","ga_20240221-135154_paper_cifar10_s2_e1_d100_40mb_div_soft","ga_20240219-085505_paper_cifar10_s2_e1_d100_40mb_div","ga_20240218-092726_paper_cifar10_s1_e1_d100_40mb_div"]
    for folder in folders:
        my_saver=Saver(folder+"test")
        my_saver.save_best_individuals_results_and_similarity(elapsed_time={20:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0},results_dir=f"/home/woody/iwb3/iwb3021h/sequence_alignment/sequence_alignment/Results/{folder}")

