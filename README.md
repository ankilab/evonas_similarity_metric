# Sequence Alignment-based Similarity Metric in Evolutionary Neural Architecture Search

## Abstract
Neural Architecture Search (NAS) has emerged as a powerful method for automating the design of deep neural networks across diverse applications, with evolutionary optimization showing particular promise in addressing its intricate demands. However, the effectiveness of this approach highly depends on balancing exploration and exploitation, ensuring that the search does not prematurely converge to suboptimal solutions while still achieving near-optimal outcomes.
This paper addresses this challenge by proposing a novel similarity metric inspired by global sequence alignment from biology. Unlike most of the existing methods that require pre-trained models for comparison, our metric operates directly on neural network architectures within the defined search space, eliminating the need for model training. We outline the computation of the normalized similarity metric and demonstrate its application in quantifying diversity within populations in evolutionary NAS. Experimental results conducted on popular datasets for image classification, such as CIFAR-10, CIFAR-100, and ImageNet16-120, show the effectiveness of our approach in guiding diversity based on our suggested control function. Additionally, we highlight the usefulness of our similarity metric in comparing individuals to identify advantageous or disadvantageous architectural design choices.

## Installation

### Setting up Virtual Environment

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
```batch

├── dashboard/
│   ├── Results/
│   ├── dashboard.py
├── datasets/
├── genetic_algorithm/
├── nnalignment/
│   ├── config.json
│   ├── example/
├── params.json
├── gene_pool.txt
├── rule_set.txt
```

## Visualize paper results
In order to plot in dash all the study results that are saved in dashboard/Results folder as .evonas files you have to run:
    ```batch
    python .\dashboard\dashboard.py
It will take almost a minute to load all the data and then a window in the browser will open. If it does not open you have to open the following URL http://127.0.0.1:8040/evo_nas/.
## Run Evolutionary NAS 

