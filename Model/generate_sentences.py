import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Generating training data or prediction data based on gene count matrix (.txt)')
parser.add_argument('--count_df_path','-c',type=str, required=True,help="The counts matrix path(.txt), rownames are genes, colnames are cell barcodes or sample names")
parser.add_argument('--length','-l',type=int, default=260,help='Length of sentence')
parser.add_argument('--save_path','-s',type=str, help='The path for saving gene sentences')

args = parser.parse_args()

def gpu_sort_index(df_path = args.count_df_path,sen_len = args.length):
    df = pd.read_table(df_path)
    df_np = df.to_numpy()
    df_np_tensor = torch.Tensor(df_np)
    df_np_tensor = df_np_tensor.to('cpu')
    sorted, indices = torch.sort(df_np_tensor, dim=0, descending=True)
    sorted_df = pd.DataFrame(sorted)
    sorted_df.columns = df.columns
    indices_tonumpy = indices.to('cpu').numpy()
    indices_tonumpy = pd.DataFrame(indices_tonumpy)
    gene_listdf = pd.DataFrame(columns = ["sentences"],index=df.columns)
    indices_tonumpy.columns = df.columns
    for i in tqdm(df.columns):
        genes = list(df.index[indices_tonumpy[i][sorted_df[i]!=0]])
        if len(genes)>sen_len:
            gene_listdf.loc[i] = " ".join(genes[1:sen_len])
        else:
            gene_listdf.loc[i] = " ".join(genes)
    return gene_listdf

gene_sentences = gpu_sort_index(df_path = args.count_df_path,sen_len = args.length)
gene_sentences['cell_barcode'] = gene_sentences.index.to_list()
gene_sentences.to_csv(args.save_path,sep="\t",index=False)