# irAEatlas
The predictive module built into irAEatlas is a knowledge-informed model that leverages single-cell transcriptomic data from patients who developed irAEs and those who did not, enabling probability prediction of irAE occurrence in patients.

Major functions: 

1.generating sentences

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
