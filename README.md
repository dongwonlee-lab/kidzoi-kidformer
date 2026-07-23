# Getting Started

## 1. Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/dongwonlee-lab/kidzoi-kidformer.git
cd kidzoi-kidformer
```

## 2. Create a Conda Environment

Create and activate a new Conda environment:

```bash
conda create --name `env_name` python=3.9
conda activate `env_name`
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
  Path to a text file containing the paths to the target **BigWig files** used for training. see `resources/targets_sum.txt`

- `genome`  
  Path to the reference genome **FASTA file**.

- `bed_file`  
  Path to the **BED file** containing the genomic sequences used for model training. see `resources/sequences_human_enformer.bed`

## Sequence Requirements

- **KidFormer** requires BED files containing **Enformer sequences**.
- **Kidzoi** requires BED files containing **Borzoi sequences**.

---

# Obtaining Variant Effect Scores

To directly obtain the `variant-effect` scores from the models directly, you can skip the training step and use the pretrained models directly. 

To obtain variant effect scores from the  models, first create the required directories:

```bash
mkdir -p resources/pretrained
mkdir -p resources/genome
```

Download the pretrained model weights from https://zenodo.org/records/19501317?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImFlZTA4Nzg4LTVhNDMtNDM5Ny1hMjE1LTFmNWE1MGQ4ZTM0MiIsImRhdGEiOnt9LCJyYW5kb20iOiJkOGFkMThmMDRmYzQ2ZmVhMmUwODQ3MGI0ZWE4MzZlZCJ9.SY1a_fB7Kn8uKNU2bDCeiW0JS-ENpjeRiwgxDp3TnsxTFHENMB2DRuoqsWETxl72U4QEFF7kRzqF0T-y2HritA and place them in:

```
resources/pretrained/
```

Place the reference genome FASTA file in:

```
resources/genome/
```

---

## Scoring with KidFormer

Run:

```bash
python ./model/scripts/score_kidformer.py \
  --vcf_file ./test.vcf \
  --output_dir ./testing_kidformer \
  --target_length 8 \
  --sad_stats SAD \
```

---

## Scoring with Kidzoi 

Run:

```bash
python ./model/scripts/score_kidzoi.py \
  --vcf_file ./test.vcf \
  --output_dir ./testing_kidzoi \
  --target_length 32 \
  --sad_stats SAD \
```

## Scoring Parameters

- `target_length`  
  Number of output bins used to compute the variant effect score.

- `sad_stats`  
  Specifies the statistics to compute:
  - `SAD`: difference between the `alt` and `ref`.


#### See `test_model_score.ipynb` for sample output from  `Kidzoi` and `Kidformer`
