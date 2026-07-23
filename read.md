# Getting Started

## 1. Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone <repository_url>
cd <repository_name>
```

## 2. Create a Conda Environment

Create and activate a new Conda environment:

```bash
conda create --name env_name python=3.9
conda activate env_name
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Preparing Training Data

Download the required training data from **GSE262931**.

After preparing the data, generate the required input files:

- **Target BigWig files** containing the experimental signals.
- **Reference genome FASTA file**.
- **BED files** containing the DNA sequences used for training.

---

# Training Models

The repository supports training both **KidFormer** (Enformer-based) and **Kidzoi** (Borzoi-based) models.

## KidFormer Training

For KidFormer, use the BED file containing **Enformer sequences**:

```bash
python ./model/utils/train_kidformer.py \
  --targets_file '/path/to/targets.txt' \
  --genome '/path/to/genome' \
  --bed-file '/path/to/enformer/sequences_human_enformer.bed'
```

## Kidzoi Training

For Kidzoi, use the BED file containing **Borzoi sequences**:

```bash
python ./model/utils/train_kidzoi.py \
  --targets_file '/path/to/targets.txt' \
  --genome '/path/to/genome' \
  --bed-file '/path/to/borzoi/sequences_human_borzoi.bed'
```

## Input Arguments

- `targets_file`  
  Path to a text file containing the paths to the target **BigWig files** used for training.

- `genome`  
  Path to the reference genome **FASTA file**.

- `bed_file`  
  Path to the **BED file** containing the genomic sequences used for model training.

## Sequence Requirements

- **KidFormer** requires BED files containing **Enformer sequences**.
- **Kidzoi** requires BED files containing **Borzoi sequences**.

---

# Obtaining Variant Effect Scores

To obtain variant effect scores from the fine-tuned models, first create the required directories:

```bash
mkdir -p resources/pretrained
mkdir -p resources/genome
```

Download the pretrained model weights and place them in:

```
resources/pretrained/
```

Place the reference genome FASTA file in:

```
resources/genome/
```

---

## Scoring with Fine-tuned KidFormer (Enformer)

Run:

```bash
python ./model/utils/score_enformer_ft.py \
  --vcf_file ./test.vcf \
  --output_dir ./out \
  --target_length 16 \
  --sad_stats SAD,logSAD \
  --shifts="-1,0,1"
```

---

## Scoring with Fine-tuned Kidzoi (Borzoi)

Run:

```bash
python ./model/utils/score_borzoi_ft.py \
  --vcf_file ./test.vcf \
  --output_dir ./out \
  --target_length 16 \
  --sad_stats SAD,logSAD \
  --shifts="-1,0,1"
```

## Scoring Parameters

- `target_length`  
  Number of output bins used to compute the variant effect score.

- `sad_stats`  
  Specifies the statistics to compute:
  - `SAD`: Sum of Absolute Differences.
  - `logSAD`: Log-transformed Sum of Absolute Differences.

- `shifts`  
  Specifies sequence shifts used during prediction to improve robustness.
