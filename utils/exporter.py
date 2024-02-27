import pandas as pd
import flammkuchen as fl
from ast import literal_eval
import json
import os
import warnings


def read_json(file_path):
  with open(file_path, 'r') as file:
      data = json.load(file)
  return data
def read_txt(file_path, literal=False):
    with open(file_path, 'r') as f:
      if literal:
        data = literal_eval(f.read()) 
      else: 
        data = f.read()
    return data
def fasta_to_dict(fasta_file_path):
    fasta_dict = {}
    current_sequence = ""
    with open(fasta_file_path, "r") as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):
                # New sequence identifier
                current_sequence = line[1:]
                fasta_dict[current_sequence] = ""
            else:
                # Append sequence data
                fasta_dict[current_sequence] += line

    return fasta_dict
################################## Merge into one file #######################
def merge_results(results_dir, study_name, export_models=False):
    try:

      df_best_individuals=pd.read_csv(f'{results_dir}/best_individuals.csv')
      sim_best_individuals=pd.read_csv(f'{results_dir}/similarity_best_individuals.csv')
    except:
      warnings.warn(f'No best individuals similarity file')
      df_best_individuals=pd.DataFrame()
      sim_best_individuals=pd.DataFrame()

    df_crossovers=pd.read_csv(f'{results_dir}/crossover_parents.csv')
    rule_set=read_txt(f'{results_dir}/rule_set.txt', literal=True)
    gene_pool=read_txt(f'{results_dir}/gene_pool.txt', literal=True)
    params=read_json(f'{results_dir}/params.json')

    results={}
    results["total"]={
        "df_best_individuals":df_best_individuals.to_json(),
        "crossover": df_crossovers.to_json(),
        "sim_best": sim_best_individuals.to_json(),
        "params": params,
        "gene_pool":gene_pool,
        "rule_set":rule_set
          }

    for gen in range(1,len(df_best_individuals)+1):
    #for gen in range(1,2):
        if os.path.exists(f'{results_dir}/Generation_{gen}'):
          try:
            alignment=read_json(f'{results_dir}/Generation_{gen}/alignment.json')
            sequences=read_txt(f'{results_dir}/Generation_{gen}/sequences.fasta')
            dict_sequences=fasta_to_dict(f'{results_dir}/Generation_{gen}/sequences.fasta')
            similarity=pd.read_csv(f'{results_dir}/Generation_{gen}/similarity.csv')
          except:
            warnings.warn(f'No similarity files Generation_{gen}')
            similarity=pd.DataFrame()
            alignment={}
            sequences=""
            dict_sequences={}

          folder_names = [folder for folder in os.listdir(f'{results_dir}/Generation_{gen}') if os.path.isdir(os.path.join(f'{results_dir}/Generation_{gen}', folder))]
          folder_names = [folder for folder in folder_names if not folder.startswith('.')]

          results[f'Generation_{gen}']={
              "alignment": alignment,
              "sequences": dict_sequences,
              "fasta_sequences": sequences,
              "similarity": similarity.to_json(),
              "individuals_names":folder_names,
          }
          individuals_results={}
          chromosomes={}
          acc_epochs={}
          for individual in folder_names[:params["population_size"]]:
            individuals_results[individual]=read_json(f'{results_dir}/Generation_{gen}/{individual}/results.json')
            chromosomes[individual]=read_json(f'{results_dir}/Generation_{gen}/{individual}/chromosome.json')
            try:
              acc_epochs[individual]=fl.load(f'{results_dir}/Generation_{gen}/{individual}/history.fl')["val_accuracy"]
            except:
              warnings.warn(f'Generation_{gen}/{individual} has no training history')
              acc_epochs[individual]= [0] * params['nb_epochs']
          results[f'Generation_{gen}']["individuals_results"]=individuals_results
          results[f'Generation_{gen}']["chromosomes"]=chromosomes
          results[f'Generation_{gen}']["acc_epochs"]=acc_epochs

    #export_folder=f'{results_dir}/compressed_results/{study_name}'
    export_folder=f'{results_dir}'
    if not os.path.exists(export_folder):
      os.makedirs(export_folder)
    with open(f"{export_folder}/results_{study_name}.evonas", 'w') as json_file:
      json.dump(results, json_file)

def save_txt(data, file_path, literal=False):
  with open(file_path, "w") as txt_file:
    if literal:
      txt_file.write(repr(data))
    else:  
      txt_file.write(data)
def save_json(data, file_path):
  with open(file_path, "w") as json_file:
    json.dump(data,json_file)

############################### Decompress results #############################
def decompress_results(results_path, export_folder, save=True):
  results=read_json(results_path)

  #Total
  df_best_individuals=pd.read_json(results["total"]["df_best_individuals"])
  df_crossovers=pd.read_json(results["total"]["crossover"])
  sim_best_individuals=pd.read_json(results["total"]["sim_best"])
  params=results["total"]["params"]
  gene_pool=results["total"]["gene_pool"]
  rule_set=results["total"]["rule_set"]

  if save:
    if not os.path.exists(export_folder):
      os.makedirs(export_folder)
    df_best_individuals.to_csv(f"{export_folder}/best_individuals.csv", index=False)
    df_crossovers.to_csv(f"{export_folder}/crossover_parents.csv", index=False)
    sim_best_individuals.to_csv(f"{export_folder}/similarity_best_individuals.csv", index=False)
    save_json(params, f"{export_folder}/params.json")
    save_txt(rule_set, f"{export_folder}/rule_set.txt", True)
    save_txt(gene_pool, f"{export_folder}/gene_pool.txt", True)

  for gen in range(1,len(df_best_individuals)+1):
      try:
        alignment=results[f'Generation_{gen}']["alignment"]
        sequences=results[f'Generation_{gen}']["fasta_sequences"]
        dict_sequences=results[f'Generation_{gen}']["sequences"]
        similarity=pd.read_json(results[f'Generation_{gen}']["similarity"])
        folder_names = results[f'Generation_{gen}']["individuals_names"]

        if save:
          if not os.path.exists(f"{export_folder}/Generation_{gen}"):
            os.makedirs(f"{export_folder}/Generation_{gen}")
          save_json(alignment, f"{export_folder}/Generation_{gen}/alignment.json")
          save_txt(sequences, f"{export_folder}/Generation_{gen}/sequences.fasta")
          similarity.to_csv(f"{export_folder}/Generation_{gen}/similarity.csv", index=False)

        individuals_results = results[f'Generation_{gen}']["individuals_results"]
        chromosomes = results[f'Generation_{gen}']["chromosomes"]
        acc_epochs = results[f'Generation_{gen}']["acc_epochs"]
        if save:
          for individual in folder_names:
            try:
              if not os.path.exists(f"{export_folder}/Generation_{gen}/{individual}"):
                os.makedirs(f"{export_folder}/Generation_{gen}/{individual}")
              save_json(individuals_results[individual],f"{export_folder}/Generation_{gen}/{individual}/results.json")
              save_json(chromosomes[individual],f"{export_folder}/Generation_{gen}/{individual}/chromosome.json")
              save_json(acc_epochs[individual],f"{export_folder}/Generation_{gen}/{individual}/val_accuracy.json")
            except:
              warnings.warn(f'Generation_{gen}/{individual} error')
      except:
        warnings.warn(f'Generation_{gen} error')




if __name__ == '__main__':
   #merge_results(results_dir, study_name, export_models=False)
   #decompress_results(results_dir, export_folder)
   #results_path=os.getcwd()
   #print(results_path)
   #files=os.listdir(results_path)
   #print(files)
   #files=[file for file in files if ".json" in file]
   #print(files)
   #print("results",f'{results_path}/{files[0]}')
   #decompress_results(f'{results_path}/{files[0]}', os.path.dirname(f'{results_path}/{files[0]}'))
   for folder in  ["/home/woody/iwb3/iwb3021h/sequence_alignment/sequence_alignment/Results/ga_20240221-142800_paper_cifar10_s4_e1_d100_40mb_div_soft","/home/woody/iwb3/iwb3021h/sequence_alignment/sequence_alignment/Results/ga_20240225-014806_paper_cifar10_s4_e1_d100_40mb_div","/home/woody/iwb3/iwb3021h/sequence_alignment/sequence_alignment/Results/ga_20240221-135154_paper_cifar10_s2_e1_d100_40mb_div_soft","/home/woody/iwb3/iwb3021h/sequence_alignment/sequence_alignment/Results/ga_20240219-085505_paper_cifar10_s2_e1_d100_40mb_div","/home/woody/iwb3/iwb3021h/sequence_alignment/sequence_alignment/Results/ga_20240218-092726_paper_cifar10_s1_e1_d100_40mb_div"]:
    #for folder in ["ga_20240130-103353_paper_imagenet_p20_s5_e10_d100"]:
      merge_results(folder, f'compressed_paper{folder.split("paper", 1)[1]}', export_models=False)