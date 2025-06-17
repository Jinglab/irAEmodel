# irAEatlas
### Graphical Abstract
![image](https://github.com/Jinglab/irAEatlas/blob/main/ForFrontPage0828.png)

### The code used in manuscript:

### Model usage: 
The predictive module built into irAEatlas is a knowledge-informed model that leverages single-cell transcriptomic data from patients who developed irAEs and those who did not, enabling probability prediction of irAE occurrence in patienxts.
#### 0.Create conda environment based on yaml file 

    conda env create -f irAEmodel_env_cuda12.4_20250617.yml 

#### 1.Generating sentences

    # Use generate_sentences.py to generate sentences in bash 
    # Help
    
    python generate_sentences.py --help
    usage: generate_sentences.py [-h] --count_df_path COUNT_DF_PATH [--length LENGTH] [--save_path SAVE_PATH]
    
    Generating training data or prediction data based on gene count matrix (.txt)
    
    optional arguments:
      -h, --help            show this help message and exit
      --count_df_path COUNT_DF_PATH, -c COUNT_DF_PATH
                            The count matrix path(.txt), rownames are genes, colnames are cell barcodes or sample names
      --length LENGTH, -l LENGTH
                            Length of sentence
      --save_path SAVE_PATH, -s SAVE_PATH
                            The path for saving gene sentences

    # Run
    python generate_sentences.py -c test_count_exp_dat.txt -l 260 -s ./sentences.txt
After running generate_sentences.py, the user will obtain a sentences.txt file containing cell-related sentences. When using this file for training, the user needs to add a column named irAE_related to the dataframe to indicate whether the cells are positively or negatively associated with the irAE phenotype.

#### 2.Training

    # Use training.py to train irAE model in bash 
    # Help
    
    python training.py --help
    usage: training.py [-h] --train_dat TRAIN_DAT [--length LENGTH] [--batch_size BATCH_SIZE] [--support SUPPORT] [--query QUERY] [--episodes EPISODES] --pretrained_model_path
                       PRETRAINED_MODEL_PATH [--save_path SAVE_PATH] [--with_gpu WITH_GPU]
    
    Training irAE model based on gene sentences from irAE positive related cells and negtive related cells
    
    optional arguments:
      -h, --help            show this help message and exit
      --train_dat TRAIN_DAT, -t TRAIN_DAT
                            The count matrix path(.txt), rownames are genes, colnames are cell barcodes or sample names
      --length LENGTH, -l LENGTH
                            Length of sentence
      --batch_size BATCH_SIZE, -b BATCH_SIZE
                            Length of sentence
      --support SUPPORT, -k SUPPORT
                            The cell numbers in support set
      --query QUERY, -q QUERY
                            The cell numbers in query set
      --episodes EPISODES, -e EPISODES
                            The training episodes
      --pretrained_model_path PRETRAINED_MODEL_PATH, -p PRETRAINED_MODEL_PATH
                            The pretrained model saved path
      --save_path SAVE_PATH, -s SAVE_PATH
                            The final model saved path
      --with_gpu WITH_GPU, -g WITH_GPU
                            Use GPU
    # Run
    python training.py -t ./train_dat.txt -l 260 -b 1 -k 2 -q 2 -e 300 -p /home/xfan/irAE/pretrained_model/ -s ./model_save_path -g True

#### 3.Predicting

    # Use predicting.py to predict irAE scores of cells or samples in bash 
    # Help
    
    python predicting.py --help
    usage: predicting.py [-h] --pred_dat PRED_DAT [--length LENGTH] --pretrained_model_path PRETRAINED_MODEL_PATH
                         [--save_checkpoint_path SAVE_CHECKPOINT_PATH] [--result_save_path RESULT_SAVE_PATH] [--with_gpu WITH_GPU]
    
    Predicting irAE scores
    
    optional arguments:
      -h, --help            show this help message and exit
      --pred_dat PRED_DAT, -i PRED_DAT
                            Prediction data file path
      --length LENGTH, -l LENGTH
                            Length of sentence
      --pretrained_model_path PRETRAINED_MODEL_PATH, -p PRETRAINED_MODEL_PATH
                            The pretrained model saved path
      --save_checkpoint_path SAVE_CHECKPOINT_PATH, -s SAVE_CHECKPOINT_PATH
                            The trained irAE model saved path
      --result_save_path RESULT_SAVE_PATH, -r RESULT_SAVE_PATH
                            The save path for the predicted results
      --with_gpu WITH_GPU, -g WITH_GPU
                            Use GPU
     # Run
     python predicting.py -i ./pred_dat.txt -l 260 -p /home/xfan/irAE/pretrained_model/ -s ./model_save_path/ -r ./ -g True

### Visitor Map

[![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=m5d_n6k5WRf2qNOJNj2u2MSngBVWuglv1yxE9N32k7Y&cl=ffffff)](https://clustrmaps.com/site/1c6oc)

     
