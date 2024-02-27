from icecream import ic
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.pairwise2 import format_alignment
import os
import json
import numpy as np
import pandas as pd
from ast import literal_eval

def load_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

config = load_config("nnalignment/config.json")

with open(config["gene_pool"], "r") as f:
    gene_pool = literal_eval(f.read())

abd="123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
one_char_dict=config["char_dict"]
# Normalization dict has the min value, max value, and max possible difference for each layer. 
normalization_dict={}
for layer in gene_pool:
    if layer["layer"] in one_char_dict.keys():
        char=one_char_dict[layer["layer"]]
        normalization_dict[char]={}
        for param, value in layer.items():
            if (param!="layer") and (param!="f_name") and (len(value)==3):
                value[2]=max(abs(value[1]-value[0]),value[2])
                #normalization_dict[char][param]=diff if diff!=0 else 1
                normalization_dict[char][param]=value
#print(normalization_dict)
def custom_match_function(char1, char2):
    #print(char1, char2)
    global dict_sequence
    global dict_sequence_2
    global translate_dict
    global translate_dict_2
    info1=dict_sequence.get(char1,{})
    info2=dict_sequence_2.get(char2,{})
    char1=translate_dict[char1]
    char2=translate_dict_2[char2]  
    ###################### P ###################
    if (char1==char2=="P"):
        norm=normalization_dict["P"]
        d=1*abs(info1["n_fft"]-info2["n_fft"])/norm["n_fft"][2]
        d+=1*abs(info1["hop_length"]-info2["hop_length"])/norm["hop_length"][2]
        #ic(3-d, "P")
        return 3-d
    elif (char1=="P") and (char2!="P"):
        return -2
    ################### C#############
    if (char1==char2=="C"):
        norm=normalization_dict["C"]
        d=abs(info1["filters"]-info2["filters"])*0.6/norm["filters"][2]
        d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/norm["kernel_size"][2]
        d+=abs(info1["strides"]-info2["strides"])*0.6/norm["strides"][2]
        #ic(3-d, "C")
        return 3-d
        
    elif (char1=="C") and (char2=="D"):
        cnorm=normalization_dict["C"]
        dnorm=normalization_dict["D"]
        norm_kernel=max(max(cnorm["kernel_size"][1]-dnorm["kernel_size"][0], dnorm["kernel_size"][1]-cnorm["kernel_size"][0]),1)
        norm_stride=max(max(cnorm["strides"][1]-dnorm["strides"][0], dnorm["strides"][1]-cnorm["strides"][0]),1)
        d=abs(info1["kernel_size"]-info2["kernel_size"])*1.2/norm_kernel
        d+=abs(info1["strides"]-info2["strides"])*0.8/norm_stride
        #ic(2-d, "CD")
        #return 2-d
        return 2-d
    elif (char1=="C") and (char2!="C"):
        return -2
    ############### D ##################
    if (char1==char2=="D"):
        norm=normalization_dict["D"]
        d=abs(info1["kernel_size"]-info2["kernel_size"])*1.2/norm["kernel_size"][2]
        d+=abs(info1["strides"]-info2["strides"])*0.8/norm["strides"][2]
        #ic(3-d, "D")
        return 3-d
    elif (char1=="D") and (char2=="C"):
        cnorm=normalization_dict["C"]
        dnorm=normalization_dict["D"]
        norm_kernel=max(max(cnorm["kernel_size"][1]-dnorm["kernel_size"][0], dnorm["kernel_size"][1]-cnorm["kernel_size"][0]),1)
        norm_stride=max(max(cnorm["strides"][1]-dnorm["strides"][0], dnorm["strides"][1]-cnorm["strides"][0]),1)

        d=abs(info1["kernel_size"]-info2["kernel_size"])*1.2/norm_kernel
        d+=abs(info1["strides"]-info2["strides"])*0.8/norm_stride
        #ic(2-2*d/3, "DC")
        #return 2-2*d/3
        return 2-d
    elif (char1=="D") and (char2!="D"):
        return -2

    ################ M ###############
    if (char1==char2=="M"):
        norm=normalization_dict["M"]
        d=abs(info1["pool_size"]-info2["pool_size"])/norm["pool_size"][2]
        #d+=abs(info1["strides"]-info2["strides"])
        #ic(3-d, "M")
        return 3-d
    elif (char1=="M") and (char2=="A"):
        mnorm=normalization_dict["M"]
        anorm=normalization_dict["A"]
        norm_pool=max(max(mnorm["pool_size"][1]-anorm["pool_size"][0], anorm["pool_size"][1]-mnorm["pool_size"][0]),1)

        d=abs(info1["pool_size"]-info2["pool_size"])/norm_pool
        #d+=abs(info1["strides"]-info2["strides"])
        return 2-d
    elif (char1=="M") and (char2!="M"):
        return -1
    ############### A ############
    if (char1==char2=="A"):
        norm=normalization_dict["A"]
        d=abs(info1["pool_size"]-info2["pool_size"])/norm["pool_size"][2]
        #d+=abs(info1["strides"]-info2["strides"])
        #ic(3-d, "A")
        return 3-d
    elif (char1=="A") and (char2=="M"):
        mnorm=normalization_dict["M"]
        anorm=normalization_dict["A"]
        norm_pool=max(max(mnorm["pool_size"][1]-anorm["pool_size"][0], anorm["pool_size"][1]-mnorm["pool_size"][0]),1)

        d=abs(info1["pool_size"]-info2["pool_size"])/norm_pool
        #d+=abs(info1["strides"]-info2["strides"])
        return 2-d
    elif (char1=="A") and (char2!="A"):
        return -1
    ############## g ################
    if (char1==char2=="g"):
        return 1.5
    elif (char1=="g") and (char2=="G"):
        return 0.5
    elif (char1=="g") and (char2!="g"):
        return -1
    ############## G #################
    if (char1==char2=="G"):
        return 1.5
    elif (char1=="G") and (char2=="g"):
        return 0.5
    elif (char1=="G") and (char2!="G"):
        return -1
    ############## E (Flatten) #################
    if (char1==char2=="E"):
        return 1.5
    elif (char1=="E") and (char2=="g"):
        return 0.5
    elif (char1=="E") and (char2=="G"):
        return 0.5
    elif (char1=="E") and (char2!="E"):
        return -1
    ############# R ###################
    if (char1==char2=="L"):
        return 1
    elif (char1=="L") and (char2!="L"):
        return -1
    ################# B ############
    if (char1==char2=="B"):
        return 2
    elif (char1=="B") and (char2=="I"):
        return 1
    elif (char1=="B") and (char2!="B"):
        return -1
    ################### I ############
    if (char1==char2=="I"):
        return 2
    elif (char1=="I") and (char2=="B"):
        return 1
    elif (char1=="I") and (char2!="I"):
        return -1
    ################### I 
    if (char1==char2=="O"):
        norm=normalization_dict["O"]
        d=2*abs(info1["rate"]-info2["rate"])/norm["rate"][2]
        return 2-d
    elif (char1=="O") and (char2!="O"):
        return -1
    ############ Downsampling #################
    if (char1==char2=="F"):
        norm=normalization_dict["F"]
        d=2*abs(info1["units"]-info2["units"])/norm["units"][2]
        #ic(3-d, "F")
        return 3-d
    elif (char1=="F") and (char2!="F"):
        return -2

    ################################
    #Residual wo connection, Residual, Bottleneck wo connection, Bottleneck
    if (char1 in {"r", "R", "t", "T"}) and (char2 in {"r", "R", "t", "T"}):
        rnorm=normalization_dict["R"]
        tnorm=normalization_dict["T"]
        norm_filters=max(max(rnorm["filters"][1]-tnorm["filters"][0], tnorm["filters"][1]-rnorm["filters"][0]),1)
        norm_kernel=max(max(rnorm["kernel_size"][1]-tnorm["kernel_size"][0], tnorm["kernel_size"][1]-rnorm["kernel_size"][0]),1)
        norm_stride=max(max(rnorm["strides"][1]-tnorm["strides"][0], tnorm["strides"][1]-rnorm["strides"][0]),1)

        if (char1==char2=='R'):
            d=abs(info1["filters"]-info2["filters"])*0.6/rnorm["filters"][2]
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/rnorm["kernel_size"][2]
            d+=abs(info1["strides"]-info2["strides"])*0.6/rnorm["strides"][2] 
            #ic(5-d, "R")   
            return 7-d
        elif (char1==char2=='T'):
            d=abs(info1["filters"]-info2["filters"])*0.6/tnorm["filters"][2]
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/tnorm["kernel_size"][2]
            d+=abs(info1["strides"]-info2["strides"])*0.6/tnorm["strides"][2]
            return 7-d
        elif (char1==char2=='r'):
            d=abs(info1["filters"]-info2["filters"])*0.6/rnorm["filters"][2]
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/rnorm["kernel_size"][2]
            d+=abs(info1["strides"]-info2["strides"])*0.6/rnorm["strides"][2]
            return 7-d
        elif (char1==char2=='t'):
            d=abs(info1["filters"]-info2["filters"])*0.6/tnorm["filters"][2]
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/tnorm["kernel_size"][2]
            d+=abs(info1["strides"]-info2["strides"])*0.6/tnorm["strides"][2]
            return 7-d
        elif (char1=="R" and char2=="T") or (char1=="T" and char2=="R"):
            d=abs(info1["filters"]-info2["filters"])*0.6/norm_filters
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/norm_kernel
            d+=abs(info1["strides"]-info2["strides"])*0.6/norm_stride
            return 5-d
        elif (char1=="R" and char2=="r") or (char1=="r" and char2=="R"):
            d=abs(info1["filters"]-info2["filters"])*0.6/rnorm["filters"][2]
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/rnorm["kernel_size"][2]
            d+=abs(info1["strides"]-info2["strides"])*0.6/rnorm["strides"][2]
            return 5-d
        elif (char1=="T" and char2=="t") or (char1=="t" and char2=="T"):
            d=abs(info1["filters"]-info2["filters"])*0.6/tnorm["filters"][2]
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/tnorm["kernel_size"][2]
            d+=abs(info1["strides"]-info2["strides"])*0.6/tnorm["strides"][2]
            return 5-d
        elif (char1=="r" and char2=="t") or (char1=="t" and char2=="r"):
            d=abs(info1["filters"]-info2["filters"])*0.6/norm_filters
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/norm_kernel
            d+=abs(info1["strides"]-info2["strides"])*0.6/norm_stride
            return 4-d
        elif (char1=="R" and char2=="t") or (char1=="t" and char2=="R"):
            d=abs(info1["filters"]-info2["filters"])*0.6/norm_filters
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/norm_kernel
            d+=abs(info1["strides"]-info2["strides"])*0.6/norm_stride
            return 3-d
        elif (char1=="T" and char2=="r") or (char1=="r" and char2=="T"):
            d=abs(info1["filters"]-info2["filters"])*0.6/norm_filters
            d+=abs(info1["kernel_size"]-info2["kernel_size"])*0.8/norm_kernel
            d+=abs(info1["strides"]-info2["strides"])*0.6/norm_stride
            return 3-d
    elif  (char1 in {"r", "R", "t", "T"}) and (char2 not in {"r", "R", "t", "T"}):
            return -2

    else:
        print(char1)
        print(char2)
        ic(0)
        return 0

def to_alignment_sequence(chromosome):
    dict_sequence={}
    translate_dict={}
    count=0
    for layer in chromosome:
          if (layer["layer"]!="MAG") and (layer["layer"]!="Rescaling"):
               if layer["layer"]=="RES_2D" or layer["layer"]=="BOT_2D":
                    if layer["skip_connection"]==0 and layer["layer"]=="RES_2D":
                         translate_dict[abd[count]]="r"
                    elif layer["skip_connection"]==0 and layer["layer"]=="BOT_2D":
                         translate_dict[abd[count]]="t"
                    else:
                         translate_dict[abd[count]]=one_char_dict[layer["layer"]]
                    dict_sequence[abd[count]]=layer
                    count+=1
               else:
                    translate_dict[abd[count]]=one_char_dict[layer["layer"]]
                    dict_sequence[abd[count]]=layer
                    count+=1
    return dict_sequence, translate_dict, abd[:count]

def sequence_to_fasta(header, sequence):
    fasta_entry = f'>{header}\n{sequence}\n'
    return fasta_entry

def chromosome_to_sequence(key, chromosome):
    sequence=""
    for layer in chromosome:
          if (layer["layer"]!="MAG") and (layer["layer"]!="Rescaling"):
               if layer["layer"]=="RES_2D" or layer["layer"]=="BOT_2D":
                    if layer["skip_connection"]==0 and layer["layer"]=="RES_2D":
                         sequence+="r"
                    elif layer["skip_connection"]==0 and layer["layer"]=="BOT_2D":
                         sequence+="t"
                    else:
                         sequence+=one_char_dict[layer["layer"]]
               else:
                    sequence+=one_char_dict[layer["layer"]]
    return sequence_to_fasta(key,sequence)

def to_original_chars(alignments, translate_dict, translate_dict_2, chrom1, chrom2):
    result_string = ""
    for char in alignments[0].seqA:
        # Use the dictionary to look up replacements, or keep the character if no replacement is defined
        try:
            replacement = translate_dict.get(char, char)
            result_string += replacement
        except:
            result_string+=char
    result_string_2 = ""
    for char in alignments[0].seqB:
        # Use the dictionary to look up replacements, or keep the character if no replacement is defined
        try:
            replacement = translate_dict_2.get(char, char)
            result_string_2 += replacement
        except:
            result_string_2+=char
    result=pairwise2.Alignment(result_string, result_string_2, alignments[0].score, alignments[0].start,alignments[0].end)
    fasta_seq1 = f">{chrom1}\n{result[0]}\n"
    fasta_seq2 = f">{chrom2}\n{result[1]}\n"
    return fasta_seq1+fasta_seq2,result[2]


def max_similarity(df):
    return np.max(df.where(np.triu(np.ones(df.shape), k=1).astype(bool)))

def control_diversity(my_ga, last_median_similarity, gen=0, soft_control=False):
    ic(last_median_similarity)
    mutation_rate=my_ga.params["mutation_rate"]
    my_ga.crossover()
    my_ga.mutation(rate=mutation_rate)
    sim_total_new_generation, _,_,_,_=align_sequences(my_ga.population_genotype)
    iterations=0
    closest_score=100
    closest_population=[]
    closest_parents=[]
    while iterations<30:
        current_med_similarity=np.median(sim_total_new_generation)
        ic(current_med_similarity)
        if soft_control:
            #if ((current_med_similarity>(last_median_similarity-max(0.1-0.1*gen/20,0.00))) & (current_med_similarity<=(last_median_similarity+max(0.1-0.1*gen/20,0.05)))):
            if ((current_med_similarity>(last_median_similarity-max(0.1-0.1*gen/20,0.00))) & (current_med_similarity<=(last_median_similarity+max(0.05*gen/20,0.025)))):
                ic("---Diversity control with both limits---")
                break
        else:
            if ((current_med_similarity>last_median_similarity) & (current_med_similarity<=(last_median_similarity+max(0.1-gen*gen/2000,0.05)))):
                break
        if abs(current_med_similarity-last_median_similarity)<closest_score:
            closest_score=abs(current_med_similarity-last_median_similarity)
            ic(closest_score)
            closest_population=my_ga.population_genotype
            closest_parents= my_ga.parents_names
        if iterations%5==0:
            mutation_rate=max(mutation_rate-2,0)
        my_ga.crossover()
        my_ga.mutation(rate=mutation_rate)
        sim_total_new_generation, _,_,_,_=align_sequences(my_ga.population_genotype)
        iterations+=1
        ic(iterations)

    if iterations==30:
        my_ga.population_genotype=closest_population
        my_ga.parents_names=closest_parents
    return my_ga.population_genotype, my_ga.parents_names


def select_similar_individuals(chromosomes,parents_names, population_size):
    sim_total_generation, _,_,_,_=align_sequences(chromosomes)

    indices_to_delete=[]
    threshold=0.5
    while (len(chromosomes)-len(indices_to_delete))>population_size:
        matrix = sim_total_generation.to_numpy()
        ic(np.median(matrix))
        upper_diagonal_mask = np.triu(np.ones(matrix.shape), k=1)
        row_indices, col_indices = np.where(np.logical_and(matrix < threshold, upper_diagonal_mask))
        if np.size(row_indices)!=0 or np.size(col_indices)!=0:
            result_df = pd.DataFrame({'Row Index': sim_total_generation.index[row_indices], 'Column Index': sim_total_generation.columns[col_indices]})
            rows_df=result_df[["Row Index"]].value_counts().reset_index()
            cols_df=result_df[["Column Index"]].value_counts().reset_index().rename(columns={"Column Index":"Row Index"})
            replace_df=pd.concat([rows_df,cols_df]).groupby("Row Index").sum().sort_values("count", ascending=False)
            med_values=sim_total_generation.median().reset_index().rename(columns={"index":"Row Index",0:"median_sim"})
            replace_df=replace_df.reset_index().merge(med_values, how="left").sort_values(by=["count","median_sim"], ascending=[False, True]).reset_index()
            ic(replace_df.head(5))
            sim_index=replace_df["Row Index"][0]
            print("Sim index: ",sim_index)
            sim_total_generation.drop(sim_index, axis=0, inplace=True)
            sim_total_generation.drop(sim_index, axis=1, inplace=True)
            indices_to_delete.append(sim_index)
        else:
            threshold+=0.05


    chromosomes = [chromosomes[i] for i in range(len(chromosomes)) if i not in indices_to_delete]
    if parents_names is None:
        return chromosomes
    else:
        parents_names = [parents_names[i] for i in range(len(parents_names)) if i not in indices_to_delete]
        return chromosomes, parents_names

def select_diverse_individuals(chromosomes,parents_names, population_size):
    sim_total_generation, _,_,_,_=align_sequences(chromosomes)

    indices_to_delete=[]
    threshold=0.7
    while (len(chromosomes)-len(indices_to_delete))>population_size:
        matrix = sim_total_generation.to_numpy()
        upper_diagonal_mask = np.triu(np.ones(matrix.shape), k=1)
        row_indices, col_indices = np.where(np.logical_and(matrix > threshold, upper_diagonal_mask))
        if np.size(row_indices)!=0 or np.size(col_indices)!=0:
            result_df = pd.DataFrame({'Row Index': sim_total_generation.index[row_indices], 'Column Index': sim_total_generation.columns[col_indices]})
            rows_df=result_df[["Row Index"]].value_counts().reset_index()
            cols_df=result_df[["Column Index"]].value_counts().reset_index().rename(columns={"Column Index":"Row Index"})
            replace_df=pd.concat([rows_df,cols_df]).groupby("Row Index").sum().sort_values("count", ascending=False)
            med_values=sim_total_generation.median().reset_index().rename(columns={"index":"Row Index",0:"median_sim"})
            replace_df=replace_df.reset_index().merge(med_values, how="left").sort_values(by=["count","median_sim"], ascending=[False, False]).reset_index()
            sim_index=replace_df["Row Index"][0]
            print("Sim index: ",sim_index)
            sim_total_generation.drop(sim_index, axis=0, inplace=True)
            sim_total_generation.drop(sim_index, axis=1, inplace=True)
            indices_to_delete.append(sim_index)
        else:
            threshold-=0.05

    sim_total_generation.to_csv(f'Results/logs/size_{str(len(chromosomes))}_out.csv')

    chromosomes = [chromosomes[i] for i in range(len(chromosomes)) if i not in indices_to_delete]
    if parents_names is None:
        return chromosomes
    else:
        parents_names = [parents_names[i] for i in range(len(parents_names)) if i not in indices_to_delete]
        return chromosomes, parents_names

def replace(chromosomes, replace_list, sim_total_generation, dict_sequences, translate_dicts, encode_sequences, max_list):
    for replace_model in replace_list:
        new_similarities=[]
        global dict_sequence
        global dict_sequence_2
        global translate_dict
        global translate_dict_2
        dict_sequences[replace_model], translate_dicts[replace_model], encode_sequences[replace_model]=to_alignment_sequence(chromosomes[replace_model])
        dict_sequence=dict_sequences[replace_model]
        translate_dict=translate_dicts[replace_model]
        seq1=encode_sequences[replace_model]
        dict_sequence_2=dict_sequences[replace_model]
        translate_dict_2=translate_dicts[replace_model]
        max_list[replace_model]=pairwise2.align.globalcs(seq1, seq1, custom_match_function, -0.5,-0.5, one_alignment_only=True)[0].score

    for replace_model in replace_list:
        new_similarities=[]
        dict_sequence=dict_sequences[replace_model]
        translate_dict=translate_dicts[replace_model]
        seq1=encode_sequences[replace_model]
        for model in range(len(chromosomes)):
            dict_sequence_2=dict_sequences[model]
            translate_dict_2=translate_dicts[model]
            seq2=encode_sequences[model]
            alignments = pairwise2.align.globalcs(seq1, seq2, custom_match_function, -0.5,-0.5, one_alignment_only=True)
            l1=len(seq1)
            l2=len(seq2)
            if max(l1,l2)==l1:
                min_score=-0.5*l1
            else:
                min_score=-0.5*l2
            max_score=max(max_list[replace_model],max_list[model])

            new_similarities.append(round((alignments[0].score-min_score)/(max_score-min_score),3))

        sim_total_generation[replace_model]=new_similarities
        sim_total_generation.loc[replace_model]=new_similarities
    return sim_total_generation, dict_sequences, translate_dicts, encode_sequences, max_list


def enforce_diversity_initial_population(my_ga, threshold=0.7):
    chromosomes=my_ga.population_genotype.copy()
    sim_total_generation, dict_sequences, translate_dicts, encode_sequences, max_list=align_sequences(chromosomes)

    iterator=0
    replace_list=["test_list"]
    last_max_similarities=len(chromosomes)
    while len(replace_list)>0:
        if iterator>200:
            break
        matrix = sim_total_generation.to_numpy()
        upper_diagonal_mask = np.triu(np.ones(matrix.shape), k=1)
        row_indices, col_indices = np.where(np.logical_and(matrix > threshold, upper_diagonal_mask))
        result_df = pd.DataFrame({'Row Index': sim_total_generation.index[row_indices], 'Column Index': sim_total_generation.columns[col_indices]})
        rows_df=result_df[["Row Index"]].value_counts().reset_index()
        cols_df=result_df[["Column Index"]].value_counts().reset_index().rename(columns={"Column Index":"Row Index"})
        replace_df=pd.concat([rows_df,cols_df]).groupby("Row Index").sum().sort_values("count", ascending=False)
        med_values=sim_total_generation.median().reset_index().rename(columns={"index":"Row Index",0:"median_sim"})
        replace_df=replace_df.reset_index().merge(med_values, how="left").sort_values(by=["count","median_sim"], ascending=[False, False])
        max_rowidx=replace_df["count"].max()
        if max_rowidx<=last_max_similarities:
            print(max_rowidx)
            if max_rowidx<10:
                replace_list=replace_df.head(1)["Row Index"].values.tolist()
            else:
                replace_list=replace_df.head(5)["Row Index"].values.tolist()
            last_max_similarities=max_rowidx
        
        if len(replace_df)==0:
            break
        
        for name in replace_list:
            chromosomes[name]=my_ga.my_gene_pool.get_random_chromosome()
        sim_total_generation, dict_sequences, translate_dicts, encode_sequences, max_list=replace(chromosomes, replace_list,sim_total_generation, dict_sequences, translate_dicts, encode_sequences, max_list)
        if max_rowidx==1:
            print(iterator)
            iterator+=1
    sim_total_generation.to_csv(f'Results/logs/initial_population_out.csv')
    return chromosomes, sim_total_generation

def align_sequences(chromosomes, results_dir=None, individuals_name=None, generation=None, from_folder=False):
    population_sequences=""
    dict_sequences, translate_dicts, encode_sequences={},{},{}
    if from_folder:
        root_folder=results_dir
        names=[f.name for f in os.scandir(f'{root_folder}/Generation_{str(generation)}') if f.is_dir()]
        chromosomes={}
        
        for name in names:
                chromosome_path=f'{root_folder}/Generation_{str(generation)}/{name}'
                # Open the JSON file for reading
                with open(f'{chromosome_path}/chromosome.json', 'r') as json_file:
                    chromosomes[name] = json.load(json_file)
    
        for key in chromosomes.keys():
            population_sequences+=chromosome_to_sequence(key, chromosomes[key])
            dict_sequences[key], translate_dicts[key], encode_sequences[key]=to_alignment_sequence(chromosomes[key])
        individuals=len(names)
        headers=individuals_name

    else:
        if results_dir or individuals_name:
            headers=individuals_name
        else:
            headers=list(np.arange(0, len(chromosomes),1))

        for idx, key in enumerate(headers):
            population_sequences+=chromosome_to_sequence(key, chromosomes[idx])
            dict_sequences[key], translate_dicts[key], encode_sequences[key]=to_alignment_sequence(chromosomes[idx])
        individuals=len(headers)

    alignments_sequences={}
    alignments_values={}
    max_list=[]
    for i in range(individuals):
            #print(i)
            global dict_sequence
            global dict_sequence_2
            global translate_dict
            global translate_dict_2
            dict_sequence=dict_sequences[headers[i]]
            translate_dict=translate_dicts[headers[i]]
            seq1=encode_sequences[headers[i]]
            dict_sequence_2=dict_sequences[headers[i]]
            translate_dict_2=translate_dicts[headers[i]]
            max_list.append(pairwise2.align.globalcs(seq1, seq1, custom_match_function, -0.5,-0.5, one_alignment_only=True)[0].score)
            
    sim_matrix=np.zeros((individuals,individuals))
    for i in range(individuals):
        dict_sequence=dict_sequences[headers[i]]
        translate_dict=translate_dicts[headers[i]]
        seq1=encode_sequences[headers[i]]
        for j in range(individuals):
            dict_sequence_2=dict_sequences[headers[j]]
            translate_dict_2=translate_dicts[headers[j]]
            seq2=encode_sequences[headers[j]]
    # Perform local sequence alignment with custom functions
    #alignments = pairwise2.align.localcc(seq1, seq2, custom_match_function, custom_gap_function,custom_gap_function, one_alignment_only=True)
            alignments = pairwise2.align.globalcs(seq1, seq2, custom_match_function, -0.5,-0.5, one_alignment_only=True)
            l1=len(seq1)
            l2=len(seq2)
            if max(l1,l2)==l1:
                min_score=-0.5*l1
            else:
                min_score=-0.5*l2
            max_score=max(max_list[i],max_list[j])

            sim_matrix[i, j]= round((alignments[0].score-min_score)/(max_score-min_score),3)
            if results_dir:
                align_sequence, original_score=to_original_chars(alignments, translate_dict, translate_dict_2, headers[i], headers[j])
                alignments_sequences[(headers[i], headers[j])]=align_sequence
                alignments_values[(headers[i], headers[j])]=round(original_score,3)
    sim_df=pd.DataFrame(sim_matrix, columns=headers[:individuals], index=headers[0:individuals])
    if results_dir:
        root_folder=str(results_dir)
        if generation is not None:
            sim_df.to_csv(f'{root_folder}/Generation_{str(generation)}/similarity.csv')
            file_path = f'{root_folder}/Generation_{str(generation)}/alignment.json'
            sequences_path = f'{root_folder}/Generation_{str(generation)}/sequences.fasta'
            #file_path_values = f'{root_folder}/Generation_{str(generation)}/alignment_score.json'
        else:
            print("Saving alignments in: ", results_dir)
            sim_df.to_csv(f'{root_folder}/similarity_best_individuals.csv')
            file_path = f'{root_folder}/best_individuals_alignment.json'
            sequences_path = f'{root_folder}/best_individuals_sequences.fasta'
        # Save the dictionary as a JSON file
        with open(file_path, "w") as json_file:
            json.dump({str(key): value for key, value in alignments_sequences.items()}, json_file)
        with open(sequences_path, 'w') as file:
            file.write(population_sequences)
    else:
        return sim_df, dict_sequences, translate_dicts, encode_sequences, max_list