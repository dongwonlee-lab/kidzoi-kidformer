
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from Bio import SeqIO
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from utils import *

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESOURCES_DIR = PROJECT_ROOT / "resources"
GENOME_PATH = RESOURCES_DIR / "genome" / "hg38.ml.fa"
CHECKPOINT_PATH = RESOURCES_DIR / "pretrained" / "kidzoi_rep0.pth"
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))


from borzoi_pytorch import Borzoi
torch.set_float32_matmul_precision('high')


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def load_finetuned_model(checkpoint_path, num_new_tracks=10, return_center_bins_only=False,bins_to_return=6144, device=None):

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
   
    pretrained = Borzoi.from_pretrained("johahi/borzoi-replicate-0",return_center_bins_only = True, bins_to_return = bins_to_return)
    model = BorzoiFineTune(pretrained, num_new_tracks=num_new_tracks)
     
    state_dict = torch.load(checkpoint_path,map_location=device)
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def main():
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--vcf_file', required=True, help='Path to VCF file')
    parser.add_argument('--shifts', type=str, default='0', help='List of shifts, e.g. --shifts -1,0,1')
    parser.add_argument('--seq_len', type=int, default=524288, help='Target length')
    parser.add_argument('--target_length', type=int, default=8, help='Target length')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--sad_stats', type=str, default='SAD', help='List of statistics to compute (e.g. SAD,REF,ALT)')
    parser.add_argument('--genome', default=str(GENOME_PATH), help='Genome FASTA')
    parser.add_argument('--targets_file', default=str(RESOURCES_DIR / 'targets_sum.txt'), help='Targets file')
    parser.add_argument('--checkpoint', default=str(CHECKPOINT_PATH), help='Model checkpoint')
    args = parser.parse_args()
    
    # Parse comma-separated strings into lists
    args.shifts = [int(x) for x in args.shifts.split(',') if x.strip()]
    args.sad_stats = [x.strip() for x in args.sad_stats.split(',') if x.strip()]
    print(f"Using shifts: {args.shifts}")
    print(f"Using stats: {args.sad_stats}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_finetuned_model(
    checkpoint_path=args.checkpoint,
    num_new_tracks=10,return_center_bins_only=True, bins_to_return=args.target_length
)

    targets_file = pd.read_csv(args.targets_file, sep='\t')
    snps = vcf_snps(args.vcf_file)
    sad_out = initialize_output_h5(args.output_dir, args.sad_stats, snps, 
                                    int(args.target_length), targets_file, args.vcf_file)

    genome_dict = SeqIO.to_dict(SeqIO.parse(args.genome, "fasta"))
    seq_len = args.seq_len
    print(f'using sequence length {seq_len}')
    dataset = VCFDataset(args.vcf_file, genome_dict, args.seq_len)
    pos_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_ref_preds, all_alt_preds = [], []
    for si, snp in tqdm(enumerate(pos_loader), total=len(pos_loader)):
        sample_ref_preds, sample_alt_preds = [], []
        
        for shift in args.shifts:
            ref_sequences, alt_sequences = dataset.get_shifted_sequences(si, shift)
            with torch.no_grad():
                ref_onehot = torch.tensor(dataset.sequence_to_onehot(ref_sequences[0])).unsqueeze(0).to(device)
                alt_onehot = torch.tensor(dataset.sequence_to_onehot(alt_sequences[0])).unsqueeze(0).to(device)
   
                ref_pred_f = model(ref_onehot.permute(0,2,1)).detach().permute(0,2,1).cpu()
                ref_pred_r = model(rev_comp(ref_onehot).permute(0,2,1)).detach().permute(0,2,1)
                ref_pred_r = torch.flip(ref_pred_r, dims=[1]).cpu()
                ref_pred = (ref_pred_f + ref_pred_r) / 2
                
                alt_pred_f = model(alt_onehot.permute(0,2,1)).detach().permute(0,2,1).cpu()
                alt_pred_r = model(rev_comp(alt_onehot).permute(0,2,1)).detach().permute(0,2,1)
                alt_pred_r = torch.flip(alt_pred_r, dims=[1]).cpu()
                alt_pred = (alt_pred_f + alt_pred_r) / 2
                
                sample_ref_preds.append(ref_pred)
                sample_alt_preds.append(alt_pred)
       
        avg_ref_pred = torch.mean(torch.stack(sample_ref_preds), dim=0).squeeze()
        avg_alt_pred = torch.mean(torch.stack(sample_alt_preds), dim=0).squeeze()
        all_ref_preds.append(avg_ref_pred)
        all_alt_preds.append(avg_alt_pred)
        write_snp(avg_ref_pred, avg_alt_pred, sad_out, si, args.sad_stats)
    sad_out.close()
    print("Done")

if __name__ == '__main__':
    main()
