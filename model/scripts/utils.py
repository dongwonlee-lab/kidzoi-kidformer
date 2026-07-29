# some of the code here is adapted from Basenji repository  https://github.com/calico/basenji


import numpy as np
import sys
import h5py
import os
import shutil
import torch
import numpy as np
import pysam
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

NUC2IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def get_lr(it, warmup_steps, max_steps):
    # 1) Linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # 2) If it >= max_steps, return min_lr
    if it >= max_steps:
        return min_lr

    # 3) Cosine decay between warmup and max_steps
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr) 


def configure_optimizers(model, weight_decay, learning_rate, device):
    # Start with all parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # Create optimizer groups
    # Any parameter that is 2D (e.g., weights in linear layers) will have weight decay
    # All others (biases, layernorms) will not
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    print(f"using fused AdamW: {use_fused}")

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=use_fused if fused_available else False
    )

    return optimizer



class BorzoiFineTune(nn.Module):
    """Simple wrapper to finetune Borzoi on new tracks"""
    def __init__(self, pretrained_model, num_new_tracks=10):
        super().__init__()
        self.borzoi = pretrained_model
        del self.borzoi.human_head
        del self.borzoi.final_softplus
        self.new_head = nn.Conv1d(1920, num_new_tracks, kernel_size=1)
        self.final_activation = nn.Softplus() 
        
    def forward(self, sequence):
        """Forward through Borzoi backbone + new head"""
        # Run through Borzoi up to final_joined_convs
        x = self.borzoi.conv_dna(sequence)
        x_unet0 = self.borzoi.res_tower(x)
        x_unet1 = self.borzoi.unet1(x_unet0)
        x = self.borzoi._max_pool(x_unet1)
        x_unet1 = self.borzoi.horizontal_conv1(x_unet1)
        x_unet0 = self.borzoi.horizontal_conv0(x_unet0)
        x = self.borzoi.transformer(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.borzoi.upsampling_unet1(x)
        x += x_unet1
        x = self.borzoi.separable1(x)
        x = self.borzoi.upsampling_unet0(x)
        x += x_unet0
        x = self.borzoi.separable0(x)
        x = self.borzoi.crop(x.permute(0, 2, 1))
        x = self.borzoi.final_joined_convs(x.permute(0, 2, 1))
        x = self.new_head(x)
        x = self.final_activation(x)  
        return x


class GenomicDataset(Dataset):
    def __init__(self, bed_file, targets_file, genome_fasta, split='train',
                 seq_len=524288, target_len=114688, bin_size=128,
                 rc_aug=False, shift_aug=False, shift_max=3):
        assert seq_len % bin_size == 0 and target_len % bin_size == 0

        # targets_sum.txt defines column order: index 0 -> col 0 (Immune), ..., 9 -> col 9 (LOH)
        targets = pd.read_csv(targets_file, sep='\t', index_col=0).sort_index()
        self.bw_paths = targets['file'].tolist()
        self.identifiers = targets['identifier'].tolist()
        self.n_targets = len(self.bw_paths)

        bed = pd.read_csv(bed_file, sep='\t', header=None,
                           names=['chrom', 'start', 'end', 'split'])
        bed = bed[bed['split'] == split].reset_index(drop=True)
        self.regions = bed[['chrom', 'start', 'end']].to_numpy()

        self.genome_fasta = genome_fasta
        self.seq_len = seq_len
        self.target_len = target_len
        self.bin_size = bin_size
        self.n_bins = target_len // bin_size

        self.rc_aug = rc_aug
        self.shift_aug = shift_aug
        self.shift_max = shift_max

        # lazily opened per worker process (pyfaidx/pyBigWig handles aren't fork-safe)
        self.genome = None
        self.bws = None

    def _init_handles(self):
        if self.genome is None:
            self.genome = Fasta(self.genome_fasta, sequence_always_upper=True, one_based_attributes=False)
        if self.bws is None:
            self.bws = [pyBigWig.open(p) for p in self.bw_paths]

    def __len__(self):
        return len(self.regions)

    @staticmethod
    def _one_hot(seq):
        seq_arr = np.frombuffer(seq.encode(), dtype=np.uint8)
        one_hot = np.zeros((4, len(seq)), dtype=np.float32)
        for nuc, idx in NUC2IDX.items():
            one_hot[idx] = seq_arr == ord(nuc)
        return one_hot

    def _clamped_window(self, chrom, center, length):
        """Return (fetch_start, fetch_end, left_pad, right_pad) for a `length`-bp
        window centered at `center`, clamped to chrom bounds."""
        chrom_len = len(self.genome[chrom])
        new_start = center - length // 2
        new_end = new_start + length
        fetch_start = max(new_start, 0)
        fetch_end = min(new_end, chrom_len)
        left_pad = fetch_start - new_start
        right_pad = new_end - fetch_end
        return fetch_start, fetch_end, left_pad, right_pad

    def __getitem__(self, idx):
        self._init_handles()
        chrom, start, end = self.regions[idx]
        center = (int(start) + int(end)) // 2

        # shift augmentation moves only the sequence window; target window stays
        # centered on the original bed midpoint, so targets are unaffected
        seq_center = center
        if self.shift_aug:
            seq_center += np.random.randint(-self.shift_max, self.shift_max + 1)

        # --- one-hot sequence over the full seq_len window ---
        seq_start, seq_end, s_lpad, s_rpad = self._clamped_window(chrom, seq_center, self.seq_len)
        seq = self.genome[chrom][seq_start:seq_end].seq
        if s_lpad or s_rpad:
            seq = 'N' * s_lpad + seq + 'N' * s_rpad
        one_hot = self._one_hot(seq)

        # --- bigwig signal over the smaller, centered target_len window ---
        t_start, t_end, t_lpad, t_rpad = self._clamped_window(chrom, center, self.target_len)

        targets = np.zeros((self.n_bins, self.n_targets), dtype=np.float32)
        for i, bw in enumerate(self.bws):
            if not (t_lpad or t_rpad):
                vals = bw.stats(chrom, t_start, t_end, nBins=self.n_bins, type="sum", exact=True)
                targets[:, i] = [v if v is not None else 0.0 for v in vals]
            else:
                # target window touches chrom edge: query only the valid sub-range at bin resolution
                bin_offset = t_lpad // self.bin_size
                n_valid_bins = (t_end - t_start) // self.bin_size
                vals = bw.stats(chrom, t_start, t_start + n_valid_bins * self.bin_size,
                                 nBins=n_valid_bins, type="sum", exact=True)
                for j, v in enumerate(vals):
                    targets[bin_offset + j, i] = v if v is not None else 0.0

        # reverse complement augmentation: reverse sequence + complement bases,
        # and reverse bin order in targets to keep them spatially aligned
        if self.rc_aug and np.random.rand() < 0.5:
            one_hot = one_hot[::-1, ::-1].copy()
            targets = targets[::-1].copy()

        return torch.from_numpy(one_hot), torch.from_numpy(targets)


def make_loader(bed_file, targets_file, genome_fasta, split='train', seq_len=524288,
                 target_len=196608,bin_size=128, batch_size=4, num_workers=8, shuffle=None,
                 rc_aug=False, shift_aug=False, shift_max=3):
    ds = GenomicDataset(bed_file, targets_file, genome_fasta, split=split,
                         seq_len=seq_len, target_len=target_len, bin_size=bin_size,
                         rc_aug=rc_aug, shift_aug=shift_aug, shift_max=shift_max)
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                       pin_memory=True, persistent_workers=(num_workers > 0))


def poisson_loss(pred, target):
    return (pred - target * torch.log(pred)).mean()


def clip_float(x, dtype=np.float16):
    return np.clip(x, np.finfo(dtype).min, np.finfo(dtype).max)


def initialize_output_h5(out_dir, sad_stats, snps, targets_length, targets_df, vcf_file):
    
    """Initialize an output HDF5 file for SAD stats."""

    os.makedirs(out_dir, exist_ok=True)
 

    shutil.copy(vcf_file, out_dir)

    num_targets = targets_df.shape[0]
    num_snps = len(snps)

    sad_out = h5py.File('%s/sad.h5' % out_dir, 'w')

    # write SNPs
    snp_ids = np.array([snp.rsid for snp in snps], 'S')
    sad_out.create_dataset('snp', data=snp_ids)

    # write SNP chr
    snp_chr = np.array([snp.chr for snp in snps], 'S')
    sad_out.create_dataset('chr', data=snp_chr)

    # write SNP pos
    snp_pos = np.array([snp.pos for snp in snps], dtype='uint32')
    sad_out.create_dataset('pos', data=snp_pos)

    # check flips
    snp_flips = [snp.flipped for snp in snps]

    # write SNP reference allele
    snp_refs = []
    snp_alts = []
    for snp in snps:
        if snp.flipped:
            snp_refs.append(snp.alt_alleles[0])
            snp_alts.append(snp.ref_allele)
        else:
            snp_refs.append(snp.ref_allele)
            snp_alts.append(snp.alt_alleles[0])
    snp_refs = np.array(snp_refs, 'S')
    snp_alts = np.array(snp_alts, 'S')
    sad_out.create_dataset('ref_allele', data=snp_refs)
    sad_out.create_dataset('alt_allele', data=snp_alts)

    # write targets
    sad_out.create_dataset('target_ids', data=np.array(targets_df.identifier, 'S'))
    sad_out.create_dataset('target_labels', data=np.array(targets_df.description, 'S'))

    # initialize SAD stats
    for sad_stat in sad_stats:
        if sad_stat in ['REF','ALT']:
            sad_out.create_dataset(sad_stat,
                shape=(num_snps, targets_length, num_targets),
                dtype='float16')
        else:            
            sad_out.create_dataset(sad_stat,
                shape=(num_snps, num_targets),
                dtype='float16')

    return sad_out



def write_snp(ref_preds, alt_preds, sed_out, xi: int, sed_stats):
    ref_preds_sum = ref_preds.sum(axis=0)
    alt_preds_sum = alt_preds.sum(axis=0)
    
    
    if 'SAD' in sed_stats:
        sed = alt_preds_sum - ref_preds_sum
        sed_out['SAD'][xi] = clip_float(sed.to(torch.float16).cpu().numpy())
    
    if 'logSAD' in sed_stats:
        log_sed = torch.log2(alt_preds_sum + 1) - torch.log2(ref_preds_sum + 1)
        sed_out['logSAD'][xi] = clip_float(log_sed.to(torch.float16).cpu().numpy())


    if 'REF' in sed_stats:
        sed_out['REF'][xi] = clip_float(ref_preds.to(torch.float16).cpu().numpy())
        
    if 'ALT' in sed_stats:
        sed_out['ALT'][xi] = clip_float(alt_preds.to(torch.float16).cpu().numpy())
       


class SNP:
    """SNP
    Represent SNPs read in from a VCF file
    Attributes:
        vcf_line (str)
    """
    def __init__(self, vcf_line, pos2=False):
        a = vcf_line.split()
        # self.chr = a[0]
        if a[0].startswith("chr"):
            self.chr = a[0]
        else:
            self.chr = "chr%s" % a[0]
        self.pos = int(a[1])
        self.rsid = a[2]
        self.ref_allele = a[3]
        self.alt_alleles = a[4].split(",")
        self.alt_allele = self.alt_alleles[0]
        self.flipped = False
        if self.rsid == ".":
            self.rsid = "%s:%d" % (self.chr, self.pos)
        self.pos2 = None
        if pos2:
            self.pos2 = int(a[5])

    def flip_alleles(self):
        """Flip reference and first alt allele."""
        assert len(self.alt_alleles) == 1
        self.ref_allele, self.alt_alleles[0] = self.alt_alleles[0], self.ref_allele
        self.alt_allele = self.alt_alleles[0]
        self.flipped = True

    def get_alleles(self):
        """Return a list of all alleles"""
        alleles = [self.ref_allele] + self.alt_alleles
        return alleles

    def indel_size(self):
        """Return the size of the indel."""
        return len(self.alt_allele) - len(self.ref_allele)

    def longest_alt(self):
        """Return the longest alt allele."""
        return max([len(al) for al in self.alt_alleles])

    def __str__(self):
        return "SNP(%s, %s:%d, %s/%s)" % (
            self.rsid,
            self.chr,
            self.pos,
            self.ref_allele,
            ",".join(self.alt_alleles),
        )










def vcf_snps(
    vcf_file,
    require_sorted=False,
    validate_ref_fasta=None,
    flip_ref=False,
    pos2=False,
    start_i=None,
    end_i=None,
):
    """Load SNPs from a VCF file"""
    if vcf_file[-3:] == ".gz":
        vcf_in = gzip.open(vcf_file, "rt")
    else:
        vcf_in = open(vcf_file)
    
    # read through header
    line = vcf_in.readline()
    while line and line[0] == "#":
        line = vcf_in.readline()
    
    # to check sorted
    if require_sorted:
        seen_chrs = set()
        prev_chr = None
        prev_pos = -1
    
    # to check reference
    if validate_ref_fasta is not None:
        import pysam
        genome_open = pysam.Fastafile(validate_ref_fasta)
    
    # read in SNPs
    snps = []
    si = 0
    while line:
        if start_i is None or (end_i is None and start_i <= si) or (start_i <= si < end_i):
            snps.append(SNP(line, pos2))
            
            if require_sorted:
                if prev_chr is not None:
                    # same chromosome
                    if prev_chr == snps[-1].chr:
                        if snps[-1].pos < prev_pos:
                            print("Sorted VCF required. Mis-ordered position: %s" % line.rstrip(), file=sys.stderr)
                            exit(1)
                    elif snps[-1].chr in seen_chrs:
                        print("Sorted VCF required. Mis-ordered chromosome: %s" % line.rstrip(), file=sys.stderr)
                        exit(1)
                seen_chrs.add(snps[-1].chr)
                prev_chr = snps[-1].chr
                prev_pos = snps[-1].pos
            
            if validate_ref_fasta is not None:
                ref_n = len(snps[-1].ref_allele)
                snp_pos = snps[-1].pos - 1
                ref_snp = genome_open.fetch(
                    snps[-1].chr, snp_pos, snp_pos + ref_n
                ).upper()
                if snps[-1].ref_allele != ref_snp:
                    if not flip_ref:
                        # bail
                        print("ERROR: %s does not match reference %s" % (snps[-1], ref_snp), file=sys.stderr)
                        exit(1)
                    else:
                        alt_n = len(snps[-1].alt_alleles[0])
                        ref_snp = genome_open.fetch(
                            snps[-1].chr, snp_pos, snp_pos + alt_n
                        ).upper()
                        # if alt matches fasta reference
                        if snps[-1].alt_alleles[0] == ref_snp:
                            # flip alleles
                            snps[-1].flip_alleles()
                        else:
                            # bail
                            print("ERROR: %s does not match reference %s" % (snps[-1], ref_snp), file=sys.stderr)
                            exit(1)
        si += 1
        line = vcf_in.readline()
    
    vcf_in.close()
    return snps



class BedDataset(Dataset):
    def __init__(
        self,
        bed_path,
        seqlen,
        genome_dict,
        core_length=None,   
        shift_aug=False,
        rc_aug=False,
        fold="train",
    ):
        self.bed_path = bed_path
        self.fold = fold

        # BED columns: chrom, start, end, target, fold
        bed_file = pd.read_csv(
            bed_path,
            sep="\t",
            names=["chrom", "start", "end", "target", "fold"],
        )

        self.bed_file = bed_file[bed_file["fold"] == fold].reset_index(drop=True)
        self.targets = self.bed_file["target"].astype(np.float32).values

        assert len(self.bed_file) == len(self.targets), \
            "Mismatch between BED rows and targets"

        self.seqlen = seqlen
        self.core_length = core_length
        self.genome_dict = genome_dict
        self.chrom_length = {c: len(genome_dict[c]) for c in genome_dict}

        if self.core_length is not None:
            assert self.core_length <= self.seqlen, \
                "core_length must be <= seqlen"

        self.shift_aug = shift_aug
        self.rc_aug = rc_aug

    
    def resize_full_interval(self, chrom, start, end):
        mid = (start + end) // 2
        ext_start = mid - self.seqlen // 2
        ext_end   = mid + self.seqlen // 2

        trimmed_start = max(0, ext_start)
        trimmed_end   = min(self.chrom_length[chrom], ext_end)

        left_pad  = trimmed_start - ext_start
        right_pad = ext_end - trimmed_end

        return trimmed_start, trimmed_end, left_pad, right_pad

    
    def resize_core_interval(self, chrom, start, end):
        mid = (start + end) // 2
        half = self.core_length // 2

        core_start = mid - half
        core_end   = mid + half

        trimmed_start = max(0, core_start)
        trimmed_end   = min(self.chrom_length[chrom], core_end)

        left_pad  = trimmed_start - core_start
        right_pad = core_end - trimmed_end

        return trimmed_start, trimmed_end, left_pad, right_pad

    
    def get_sequence(self, chrom, start, end):

       
        if self.core_length is None:
            ts, te, lp, rp = self.resize_full_interval(chrom, start, end)
            seq = str(self.genome_dict[chrom].seq[ts:te]).upper()
            return ('N' * lp) + seq + ('N' * rp)

       
        ts, te, lp, rp = self.resize_core_interval(chrom, start, end)
        core_seq = str(self.genome_dict[chrom].seq[ts:te]).upper()
        core_seq = ('N' * lp) + core_seq + ('N' * rp)

        # Pad to full seqlen
        total_pad = self.seqlen - len(core_seq)
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad

        return ('N' * left_pad) + core_seq + ('N' * right_pad)

   
    def sequence_to_onehot(self, sequence):
        mapping = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0],
        }
        return np.array(
            [mapping.get(b, [0, 0, 0, 0]) for b in sequence],
            dtype=np.float32,
        )

    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        row = self.bed_file.iloc[idx]
        chrom = row["chrom"]
        start = int(row["start"])
        end   = int(row["end"])

        target = np.array([self.targets[idx]], dtype=np.float32)

       
        if self.shift_aug:
            shift = np.random.randint(-3, 4)
            start += shift
            end   += shift

        sequence = self.get_sequence(chrom, start, end)
        onehot = self.sequence_to_onehot(sequence)

        # Optional reverse-complement
        if self.rc_aug and np.random.rand() < 0.5:
            onehot = onehot[::-1, ::-1].copy()

        return {
            "sequence": torch.from_numpy(onehot),  
            "target": torch.from_numpy(target),
        }


class VCFDataset(Dataset):
    
    def __init__(self, file_path, genome_dict, seqlen):
        self.file_path = file_path
        self.vcf_file = pd.read_csv(file_path, sep='\t', header=None, comment = '#', usecols=range(5))
        self.vcf_file.columns = ['chrom', 'pos', 'id', 'ref', 'alt']
        self.seqlen = seqlen
        self.genome_dict = genome_dict
        self.chrom_length = {chrom: len(genome_dict[chrom]) for chrom in genome_dict}
        
    def resize_interval(self, chrom, start, end):
        mid_point = (start + end) // 2
        extend_start = mid_point - self.seqlen // 2
        extend_end = mid_point + self.seqlen // 2
        trimmed_start = max(0, extend_start)
        left_pad = trimmed_start - extend_start
        trimmed_end = min(self.chrom_length[chrom], extend_end)
        right_pad = extend_end - trimmed_end
        return trimmed_start, trimmed_end, left_pad, right_pad
        
    def get_sequence(self, chrom, start, end, shift=0):
        if shift:
            start += shift
            end += shift
        trimmed_start, trimmed_end, left_pad, right_pad = self.resize_interval(chrom, start, end)
        sequence = str(self.genome_dict[chrom].seq[trimmed_start:trimmed_end]).upper()
        left_pad_seq = 'N' * left_pad
        right_pad_seq = 'N' * right_pad
        sequence = left_pad_seq + sequence + right_pad_seq
        return sequence
        
    def sequence_to_onehot(self, sequence):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0],
                   'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        onehot = np.array([mapping[base] for base in sequence], dtype=np.float32)
        return onehot
    
    def __len__(self):
        return len(self.vcf_file)
        
    def __getitem__(self, idx):
        chrom, pos, ref, alt = self.vcf_file.loc[idx, ['chrom', 'pos', 'ref', 'alt']]
        pos = pos - 1
        if not str(chrom).startswith("chr"):
            chrom = f"chr{chrom}"
        ref_sequence = self.get_sequence(chrom, pos, pos + 1, shift=0)
        assert ref_sequence[self.seqlen // 2] == ref
        alt_sequence = ref_sequence[:self.seqlen // 2] + alt + ref_sequence[self.seqlen // 2 + 1:]
        return {"ref_sequence": ref_sequence, "alt_sequence": alt_sequence,
                "chrom": chrom, "pos": pos + 1, "ref": ref, "alt": alt}
    
    def get_shifted_sequences(self, idx, shift):
        chrom, pos, ref, alt = self.vcf_file.loc[idx, ['chrom', 'pos', 'ref', 'alt']]
        pos = pos - 1
        if not str(chrom).startswith("chr"):
            chrom = f"chr{chrom}"
        ref_sequences, alt_sequences = [], []
        ref_seq = self.get_sequence(chrom, pos, pos + 1, shift=shift)
        center_idx = self.seqlen // 2 
        original_pos_in_seq = center_idx - shift
        if 0 <= original_pos_in_seq < len(ref_seq):
            alt_seq = ref_seq[:original_pos_in_seq] + alt + ref_seq[original_pos_in_seq + 1:]
        else:
            raise ValueError(f"Ref mismatch at {chrom}:{pos+1}, shift={shift}. Expected '{ref}', got '{ref_seq[original_pos_in_seq]}'")
        ref_sequences.append(ref_seq)
        alt_sequences.append(alt_seq)
        return ref_sequences, alt_sequences

def rev_comp(snp: torch.Tensor) -> torch.Tensor:
    if snp.dim() == 2:
        rc = snp.flip(0)[:, [3, 2, 1, 0]]
    elif snp.dim() == 3:
        rc = snp.flip(1)[:, :, [3, 2, 1, 0]]
    else:
        raise ValueError("Input must have shape (L, 4) or (B, L, 4)")
    return rc











