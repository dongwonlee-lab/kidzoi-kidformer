import sys
import torch
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import *
from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper
import random


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

#torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('high')

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESOURCES_DIR = PROJECT_ROOT / "resources"
GENOME_PATH = RESOURCES_DIR / "genome" / "hg38.ml.fa"
DEFAULT_CHECKPOINT_PATH = RESOURCES_DIR / "pretrained" / "kidformer.pth"
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))


def main():
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent

    parser = argparse.ArgumentParser()
    parser.add_argument('--vcf_file', required=True, help='Path to VCF file')
    parser.add_argument('--shifts', type=str, default='0', help='List of shifts, e.g. --shifts -1,0,1')
    parser.add_argument('--seq_len', type=int, default=196608, help='Target length')
    parser.add_argument('--target_length', type=int, default=8, help='Target length')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--sad_stats', default='SAD', help='List of statistics to compute (e.g. SAD,logSAD,REF,ALT)')
    parser.add_argument('--genome', default=str(GENOME_PATH), help='Genome FASTA')
    parser.add_argument('--targets_file', default=str(RESOURCES_DIR / 'targets_sum.txt'), help='Targets file')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT_PATH), help='Model checkpoint')
    args = parser.parse_args()

    args.shifts = [int(x) for x in args.shifts.split(',') if x.strip()]
    args.sad_stats = [x.strip() for x in args.sad_stats.split(',') if x.strip()]
    print(f"Using shifts: {args.shifts}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    pretrained = from_pretrained('EleutherAI/enformer-official-rough', target_length=args.target_length)
    model = HeadAdapterWrapper(enformer=pretrained, num_tracks=10, post_transformer_embed=False).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt if not isinstance(ckpt, dict) else ckpt
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model = model.eval()
    print("Checkpoint loaded")
    print(f'using sequence length {args.seq_len}')

    targets_file = pd.read_csv(args.targets_file, sep='\t')
    
    snps = vcf_snps(args.vcf_file)
    sad_out = initialize_output_h5(args.output_dir, args.sad_stats, snps, 
                                    int(args.target_length), targets_file, args.vcf_file)

    genome_dict = SeqIO.to_dict(SeqIO.parse(args.genome, "fasta"))
    seq_len = args.seq_len
    dataset = VCFDataset(args.vcf_file, genome_dict, seq_len)
    pos_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_ref_preds, all_alt_preds = [], []
    for si, snp in tqdm(enumerate(pos_loader), total=len(pos_loader)):
        sample_ref_preds, sample_alt_preds = [], []
        
        for shift in args.shifts:
            ref_sequences, alt_sequences = dataset.get_shifted_sequences(si, shift)
            with torch.no_grad():
                ref_onehot = torch.tensor(dataset.sequence_to_onehot(ref_sequences[0])).unsqueeze(0).to(device)
                alt_onehot = torch.tensor(dataset.sequence_to_onehot(alt_sequences[0])).unsqueeze(0).to(device)
                ref_pred_f = model(ref_onehot).detach().cpu()
                ref_pred_r = model(rev_comp(ref_onehot))
                ref_pred_r = torch.flip(ref_pred_r, dims=[1]).detach().cpu()  
                ref_pred = (ref_pred_f + ref_pred_r) / 2
                alt_pred_f = model(alt_onehot)
                alt_pred_r = model(rev_comp(alt_onehot)).detach().cpu()  
                alt_pred_r = torch.flip(alt_pred_r, dims=[1]).detach().cpu()  
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
