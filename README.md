# The predictive model of irAEs
### Distinct immune cell dynamics associated with immune-related adverse events during combined chemoradiation and immune checkpoint inhibitor therapy
### Lei Zhang, Xiaokai Fan, Jun Ma, Jun Zhang , Ying Wei, Bin Hu, Di Zhou, Junjun Zhou, Yongrui Bai, Jianming Tang, Xiumei Ma, Haiyan Chen, Ying Jing
In this study, we systematically characterized immune cells in peripheral blood and tumor tissues throughout combined chemoradiotherapy and immune checkpoint inhibitor (ICI) treatment using single-cell RNA sequencing (scRNA-seq) and single-cell V(D)J sequencing (scVDJ-seq). We found that chemoradiotherapy and ICI exert distinct effects on various immune cell populations associated with immune-related adverse events (irAEs). To capture the diverse roles and interactions of these cells, we employed deep learning to extract representative molecular features and develop predictive models for irAEs. These models were built using scRNA-seq data from blood samples collected at three time points: pre-radiotherapy (Model 1), pre-immunotherapy (Model 2), and on-immunotherapy (Model 3). Immune cell subclusters at each time point were classified as irAE-related—significantly enriched in irAE patients—or non-irAE-related—significantly enriched in non-irAE patients. Validation across multiple independent cohorts demonstrated that Model 3 achieved the highest accuracy in predicting irAE occurrence. The code used to develop these models is provided here. For detailed methods and model descriptions, please refer to our paper.

### Usage: 
Before you run the pipeline, please install miniconda to manage software and dependencies.
#### 0.Create conda environment based on yaml file 
    conda env create -f irAEmodel_env_cuda12.4_20250617.yml 
Once the conda environment is set up, the user can activate the environment by

    conda activate iraemodel 
    
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
After running generate_sentences.py, the user will obtain a sentences.txt file containing cell-related sentences. For training purposes, the user needs to add a column named irAE_related to the dataframe. This column should indicate whether the cells are positively or negatively associated with the irAE phenotype.
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

The training.py script can be used to train the irAE model. Once trained, the model can be applied to predict irAE scores for individual cells or samples using the predicting.py script.

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

### Visitor map:

[![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=m5d_n6k5WRf2qNOJNj2u2MSngBVWuglv1yxE9N32k7Y&cl=ffffff)](https://clustrmaps.com/site/1c6oc)

     
