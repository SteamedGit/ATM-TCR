import argparse
import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from attention import Net
from data_loader import define_dataloader, load_embedding, load_data_split
from utils import str2bool, timeSince, get_performance_batchiter, print_performance, write_blackbox_output_batchiter



def main():
    parser = argparse.ArgumentParser(description='Prediction of TCR binding to peptide-MHC complexes')
    parser.add_argument('--indepfile', type=str, default=None,
                        help='Independent test data file',required=True)
    parser.add_argument('--blosum', type=str, default=None,
                        help='File containing BLOSUM matrix to initialize embeddings')
    parser.add_argument('--model_name', type=str, default='models/original.ckpt',
                        help = 'Model name and path to be loaded for testing ')
    parser.add_argument('--results_dir',type=str,default="result/",help='Path to results dir')
    parser.add_argument('--cuda', type=str2bool, default=True,
                        help = 'enable cuda')
    parser.add_argument('--drop_rate', type=float, default=0,
                        help='dropout rate')
    parser.add_argument('--seed', type=int, default=1039,
                        help='random seed')
    parser.add_argument('--lin_size', type=int, default=1024,
                        help='size of linear transformations')
    parser.add_argument('--padding', type=str, default='mid',
                        help='front, end, mid, alignment')
    parser.add_argument('--heads', type=int, default=5,
                        help='Multihead attention head')
    parser.add_argument('--max_len_tcr', type=int, default=20,
                        help='maximum TCR length allowed')
    parser.add_argument('--max_len_pep', type=int, default=22,
                        help='maximum peptide length allowed')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load embedding matrix
    embedding_matrix = load_embedding(args.blosum)
    eval_df = pd.read_csv(args.indepfile)
    eval_df.columns = ['peptide', 'cdr3b', 'binder'] 
    peptides = eval_df['peptide'].unique() 

    peptide_dfs = {}  # Create an empty dictionary to store DataFrames
    
    model = Net(embedding_matrix, args).to(device)
    model.load_state_dict(torch.load(args.model_name, map_location=torch.device('cpu')))
    if args.results_dir not in os.listdir('.'):
        os.makedirs(args.results_dir,exist_ok=True)

    pep_auc = {}
    model.eval()
    for peptide in peptides:
        pep_df = eval_df[eval_df['peptide'] == peptide].copy()

        indep_loader = define_dataloader(pep_df['peptide'].to_numpy(),
                                         pep_df['cdr3b'].to_numpy(),
                                         pep_df['binder'].to_numpy(),
                                         maxlen_pep=args.max_len_pep, 
                                         maxlen_tcr=args.max_len_tcr,
                                         padding=args.padding,
                                         batch_size=1,
                                         device=device)


     
        perf_indep = get_performance_batchiter(
            indep_loader['loader'], model, device,return_scores=True)
        print(f'{peptide} ----------------')
        print(perf_indep['auc'])
        pep_auc[peptide]=perf_indep['auc']
        pep_df['score'] = [x[0] for x in perf_indep['score']]
        pep_df.to_csv(f'{args.results_dir}/{peptide}_scores.csv',index=False)



        peptide_dfs[peptide] = pep_df
    print(pep_auc)

if __name__=="__main__":
    main()