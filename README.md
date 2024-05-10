# Spiking topic Model (STM)

This repository contains source code of STM model presented in [publication link]. STM is spiking neural network topic model than can effectively detect topics from the corpus of documents. All the experiments mentioned in the publication can be find in this repository.


## Folder Structure 

├── data (**dataset folder**)\
│   ├── 20news\
│   ├── 20newstest\
│   ├── 20newstrain\
│   ├── AG\
│   ├── bbc\
│   └── stopwords.txt\
├── model (**python source code of the model and evaluation**)\
├── model-input-data (**preprocessed datasets**)\
│   ├── 20news\
│   ├── ag\
│   └── bbc\
├── model-output-data (**trained models**)\
│   ├── 20news\
│   ├── ag\
│   ├── bbc\
├── network-configuration (**Brain2 physical parameters of the STM**)\
│   ├── encoder_inhibitory_synapses.yaml\
│   ├── encoder_excitatory_layer.yaml\
│   ├── encoder_excitatory_synapses.yaml\
├── IR_PURITY_eval.py (**Purity and IR evaluation**)\
├── LDA_model_script.py (**LDA model runner**)\
├── requirements.txt\
├── STM-model-script.py (**STM model runner**)\
├── TOPIC_eval.py (**Topic evaluation**)\
├── BTM_model_script.py (**BTM model runner**)\

## STM model description

The STM model is implemented with usage of  [brian2](https://brian2.readthedocs.io/en/stable/) SNN simulator. 
Spiking neural network physical parameters and differential equations are in the `/network-configuration` folder. The main class of the STM model is in the `/model/network/stm_model_runner.py` file. The class loads the configuration from the yaml files and composes the network. Some paramters from the yaml file are overwritten. They can be accessed directly from the STMModelRunner class. Please check `./STM-model-script.py` for more details.

## Running models

STM, Latent Dirichlelt Allocation and Biterm models can be run from runner scripts e.g STM-model-script.py, LDA_model_script.py, BTM_model_script.py.

The ETM experiments can be found in seperate repo https://github.com/mmarcinmichal/stm-ssn-topic-model-etm-part.

All the scripts are run for the particular data set. In the first step script checks if the dataset was preprocessed before. The preprocessed dataset is located in the `\model-input-data folder`. Suppose the dataset is processed for the first time; in such case, the script will load data from the `\data` folder, preprocess the dataset, and save it in the `\model-input-data` folder. Custom datasets can be added to `\data` folder. To use them, a new data set loader needs to be implemented. Please check the `\model\datasets` source code for more information.

When the preprocessing is completed, the script will train the model base on the data from `\model-input-data`. The model after training will be saved in the `\model-output-data`. 

The runner script will also compute topic representation for the dataset. The topic representation will be saved along with the model in the  `\model-output-data` folder. 


## Topic evaluation

 Each model script evaluates topic with usage of [Palmetto](https://github.com/dice-group/Palmetto), we hardcoded the publicly avaialble Palemtto endpoint, however for more intense evaluation we recomend use local instance of Palmetto.

There is also `TOPIC_eval.py` script which can be used to evaluate topics.
`TOPIC_eval.py` loads the models from the model files located in the `\model-output-data`. After that, it extracts the topics and performs an evaluation. To calculate topic metrics using [Palmetto](https://github.com/dice-group/Palmetto) library, `ENDPOINT_PALMETTO` variable need to be set. In our experiments, we run Palmetto docker. Please refer to the documentation of the  [Palmetto](https://github.com/dice-group/Palmetto) library for more details. If `ENDPOINT_PALMETTO` points Palmetto service, the script will extract the topics from the models and calculate coherence metrics. The result of the evaluation will be saved `\model-output-folder`.

    
## Purity and Information Retrival evaluation

Analogically cluster purity and information retrieval is performed. `IR_PURITY_eval.py` script loads the topic-based representation of the dataset from the  `\model-output-folder` for each model than purity and fscore are calculated. After that, the script saves the result in the `\model-output-folder`.


## GPU implementation (STMG)

The STMG-model-script.py is a GPU-version of STM method. 