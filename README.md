# Sequence Alignment-based Similarity Metric in Evolutionary Neural Architecture Search

## Abstract
Neural Architecture Search (NAS) has emerged as a powerful method for automating the design of deep neural networks across diverse applications, with evolutionary optimization showing particular promise in addressing its intricate demands. However, the effectiveness of this approach highly depends on balancing exploration and exploitation, ensuring that the search does not prematurely converge to suboptimal solutions while still achieving near-optimal outcomes.
This paper addresses this challenge by proposing a novel similarity metric inspired by global sequence alignment from biology. Unlike most of the existing methods that require pre-trained models for comparison, our metric operates directly on neural network architectures within the defined search space, eliminating the need for model training. We outline the computation of the normalized similarity metric and demonstrate its application in quantifying diversity within populations in evolutionary NAS. Experimental results conducted on popular datasets for image classification, such as CIFAR-10, CIFAR-100, and ImageNet16-120, show the effectiveness of our approach in guiding diversity based on our suggested control function. Additionally, we highlight the usefulness of our similarity metric in comparing individuals to identify advantageous or disadvantageous architectural design choices.

## Installation


### Setting up Virtual Environment
Python >= 3.10.11 is required for this repository
#### Windows

1. Open Command Prompt (`cmd`) or Git Bash.
2. Navigate to your project directory.
3. Create a virtual environment:
   ```batch
   python -m venv .venv
4. Activate the virtual environment:
   ```batch
   .venv\Scripts\activate
5. Install dependencies from requirements.txt:
   ```batch
   pip install -r requirements.txt

#### Linux

1. Open Terminal.
2. Navigate to your project directory.
3. Create a virtual environment:
   ```batch
   python3 -m venv .venv
4. Activate the virtual environment:
   ```batch
   source .venv/bin/activate
5. Install dependencies from requirements.txt:
   ```batch
   pip install -r requirements.txt

You're ready to use the project within the virtual environment.

## Code structure
Our repository is divided in 3 main sections: 
1. **Dashboard**
This section is dedicated to visualizing results.

**Results** Contains saved paper results and  **dashboard.py** the Script for running the dashboard.
```batch
├── dashboard/
│   ├── Results/
│   ├── dashboard.py
```
**2. Datasets and Genetic Algorithm**
This section handles datasets used in the experiments comprises genetic algorithm functions for running evolutionary NAS from main.py.
```
├── datasets/
├── genetic_algorithm/
├── main.py
```

**3. nnalignment**
Handles computation of similarity scores, which uses its config.json file to load the gene_pool.txt that defines our search space. 
```batch
├── nnalignment/
│   ├── config.json
│   ├── example/
```
rule_set.txt defines individuals macrostructure and params.json the dataset (CIFAR10,CIFAR100,IMAGENET16-120) and other parameters to run main.py in NAS algorithm.
```batch
├── params.json
├── gene_pool.txt
├── rule_set.txt
```
## Visualizing Paper Results
To visualize the results presented in the paper, which are saved in dashboard/Results folder as .evonas files, execute the following command in your terminal:
   ```batch
   python .\dashboard\dashboard.py
   ```
This process may take nearly a minute to load all the data. Once completed, a new browser window will open automatically. If it doesn't, you can access the dashboard via the URL http://127.0.0.1:8040/evo_nas/. In this dashboard, you can compare the evolution of fitness and similarity between two studies. Additionally, you can analyze detailed similarity matrices and the evolution of diversity within one of the studies. 

Each test is named according to the dataset, followed by the seed number, and whether diversity control is applied. For instance, a test for CIFAR100 with seed 1 is denoted as *cifar100_s1*, while one with diversity control is labeled as *cifar100_s1_div*.
## Run Evolutionary NAS 
If you wish to conduct a new study on Evolutionary NAS or replicate one of the existing ones, follow these instructions:

1. **Modify Parameters**: Adjust the parameters in the **params.json** file according to your requirements.
2. **Execute the Command**: Run the following command in your terminal:
   ```batch
   python main.py
   ```
Ensure that you set **gpu=True** in the params file if you are using GPU and adjust the number of parallel_processes to optimize training times. For instance, in our experiments, we utilized an A100 80GB GPU and trained 5 models simultaneously.After the algorithm completes its execution, it generates an *.evonas* file within the Results folder corresponding to the study. To visualize these results, it is essential to manually transfer this file to the **dashboard/Results folder**.

3. **Linux Compatibility**: You can also create or run scripts in Linux with predefined parameters. For example, to execute CIFAR-10 with rigid diversity control and seed 1, use the following script:
   ```batch
   ./scripts/run_evo_nas_cifar10_seed_1_div.sh
   ```
To execute Naive CIFAR-10 and seed 1, use the following script:
   ```batch
   ./scripts/run_evo_nas_cifar10_seed_1_naive.sh
   ```
To execute CIFAR-10 and seed 1 with soft diversity control, use the following script:
   ```batch
   ./scripts/run_evo_nas_cifar10_seed_1_div_soft.sh
   ```
You can create new scripts or modify existing ones to run experiments with CIFAR100 and IMAGENET16-120 datasets as well. Ensure the parameters are appropriately set for your experiments.

By following these steps, you can effectively visualize our paper results and conduct new studies on Evolutionary NAS.