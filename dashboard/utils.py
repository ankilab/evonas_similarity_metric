import pandas as pd
from difflib import SequenceMatcher
from suffix_tree import Tree
import json
import numpy as np
from io import StringIO

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
def dict_to_fasta(input_dict):
    fasta_lines = []
    for name, sequence in input_dict.items():
        fasta_lines.append(f'>{name}')
        fasta_lines.append(sequence)

    fasta_content = '\n'.join(fasta_lines)
    return fasta_content
def string_diff(str1, str2):
    matcher = SequenceMatcher(None, str1, str2)
    diffs = list(matcher.get_opcodes())
    subs_list = []
    del_list = []
    ins_list = []
    for tag, i1, i2, j1, j2 in diffs:
        if tag == 'replace':
            subs_list.append({"value": f'{str1[i1:i2]}->{str2[j1:j2]}',
                              "good": str1[i1:i2],
                              "wrong": str2[j1:j2],
                              "good_position":f"{i1}:{i2-1}",
                              "wrong_position":f"{j1}:{j2-1}"})
        elif tag == 'delete':
            del_list.append({"value":str1[i1:i2],
                             "positions":f"{i1}:{i2-1}"})
        elif tag == 'insert':
            ins_list.append({"value":str2[j1:j2],
                             "positions":j1})
    return subs_list, del_list, ins_list
def read_json(file_path):
  with open(file_path, 'r') as file:
      data = json.load(file)
  return data

def find_substring_positions(dictionary, substring):
    positions = {}
    for key, value in dictionary.items():
        positions[key] = [i for i in range(len(value)) if value.startswith(substring, i)]
    return positions
def get_lists(file_path, test_name):
    results=read_json(file_path)
    ngens=len(results.keys())-1
    ####################### Best individuals ###################
    best_individuals=pd.read_json(StringIO(results["total"]["df_best_individuals"]))
    best_sim_df=pd.read_json(StringIO(results["total"]["sim_best"]))

    best_individuals_sequences_dict={}
    for individual in best_individuals.Name:
            best_individuals_sequences_dict[individual]=results[f'Generation_{best_individuals[best_individuals.Name==individual]["Generation"].item()}']["sequences"][individual]
    best_individuals_sequences=dict_to_fasta(best_individuals_sequences_dict)
    ######################### Good cases ################################
    fitness_scores = best_individuals.set_index('Name')['fitness'].to_dict()
    tree=Tree(best_individuals_sequences_dict)

    repeated_substrings=[]
    for C, path in sorted(tree.maximal_repeats(), reverse=True):
        if C>=4:
            repeated_substrings.append(str(path).replace(" ", ""))

    dict_substrings={}
    for substring_to_find in repeated_substrings[:]:
        len_list=[]
        list_positions=[]
        counter=0
        total_weight=0
        positions = find_substring_positions(best_individuals_sequences_dict, substring_to_find)
        for individual, position in positions.items():
            if len(position)!=0:
                total_weight+=fitness_scores[individual]
                len_list.append(len(position))
                counter+=1
                list_positions.extend(position)
        med_value=int(np.median(len_list))
        typical_positions=[f"{str(int(np.percentile(list_positions,i*100/(med_value+1))))};" for i in range(1, med_value+1)]
        dict_substrings[substring_to_find]={"appearances":counter,
                                            "weight":round(total_weight,2),
                                            "median_repetitions":med_value,
                                            "typical_positions":typical_positions}

    df_good_cases_best_layers = pd.DataFrame.from_dict(dict_substrings, orient='index').reset_index().rename(columns={"index":"layers"})
    ######################## Parameters #########################
    parameters=results["total"]["params"]
    ########################## Similarity matrix ###################
    df_list= [pd.read_json( StringIO(results[f'Generation_{gen}']["similarity"])).rename(columns={"Unnamed: 0":"index"}).set_index("index") for gen in range(1,ngens+1)]
    ########################### Alignments ###############
    alignment_dict=[results[f'Generation_{gen}']["alignment"] for gen in range(1,ngens+1)]
    ########################### val accuracy / epochs ###################
    val_acc_list = [results[f'Generation_{gen}']["acc_epochs"] for gen in range(1,ngens+1)]
    ########################### Results ################################    
    results_list=[results[f'Generation_{gen}']["individuals_results"] for gen in range(1,ngens+1)]
    ###########################
    df_results=[pd.DataFrame.from_dict(results_list[gen-1],orient='index').reset_index().rename(columns={"index":"name"}).assign(Generation=gen) for gen in range(1,ngens+1)]

    ############ Get wrong layers patterns #########
    wrong_cases_total=[]
    for gen in range(1,len(df_list)+1):
        sequences_gen=results[f'Generation_{gen}']["sequences"]
        row_indices, col_indices = np.where((df_list[gen-1] > 0.8) & (df_list[gen-1] != 1))
        selected_indices = list(zip(df_list[gen-1].index[row_indices], df_list[gen-1].columns[col_indices]))
        for index in selected_indices:
            fitness_diff=results_list[gen-1][index[0]]["fitness"]-results_list[gen-1][index[1]]["fitness"]
            if (fitness_diff > 0.1) & (fitness_diff < 1):
                sim_score=df_list[gen-1].at[index[0], index[1]]
                subs,delet,ins=string_diff(sequences_gen[index[0]],sequences_gen[index[1]])
                wrong_cases_total.append({"generation":gen,"individuals":index,"sequences":[sequences_gen[index[0]],sequences_gen[index[1]]] ,"difference":round(fitness_diff,3),"similarity":round(sim_score,3),"substitutions":subs,"deletions":delet, "insertions":ins})
    #with open(f'Design/{test_name}_bad_design_total.json', 'w') as json_file:
    #    json.dump(wrong_cases_total, json_file, indent=2)
    design_dataframe=[]
    for change in wrong_cases_total:
        design_dataframe.append([change["generation"],change["individuals"][0],change["individuals"][1],change["sequences"][0],change["sequences"][1], change["difference"], change["similarity"],[f'{subs["value"]};' for subs in change["substitutions"]], [f'{ins["value"]};' for ins in change["insertions"]], [f'{delet["value"]};' for delet in change["deletions"]]])
    bad_design_df=pd.DataFrame(design_dataframe, columns=["Generation","Individual_1","Individual_2", "Sequence_1", "Sequence_2", "Fitness difference", "Similarity","Substitutions","Insertions", "Deletions"]).sort_values(by=["Similarity","Fitness difference"], ascending=[False, False]).reset_index(drop=True)
    return df_list,  alignment_dict, results_list, df_results, val_acc_list, parameters, best_individuals, best_sim_df,  best_individuals_sequences,bad_design_df, df_good_cases_best_layers
