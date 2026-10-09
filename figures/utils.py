import pandas as pd
import numpy as np
import torch
import warnings

from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import matplotlib.pyplot as plt
import seaborn as sns

#utility functions

def compute_cell_specific_correlations(
    target_dir,
    pred_dir,
    cells_bed,
    cells_pred,
    n_cell_ids=10,
    value_threshold=1,
    pred_slice=(447, 449),
    verbose=True,
):
   

    corrs_df = pd.DataFrame(
        index=cells_bed,
        columns=[f'cell_{i}' for i in range(n_cell_ids)],dtype=float
    )

    for bed_cell, pred_cell in zip(cells_bed, cells_pred):
        # Load targets
        targets_df = pd.read_csv(
            f"{target_dir}/{bed_cell}.bed",
            sep="\t",
            names=['chrom', 'start', 'end', 'value', 'fold']
        )
        # Load predictions
        predictions = torch.load(f"{pred_dir}/{pred_cell}.pt").permute(0, 2, 1)
        
        # Filter test fold and apply threshold
        value = targets_df.loc[targets_df['fold'] == 'test', 'value']
        value_tensor = torch.tensor(value.values)
        mask = value_tensor > value_threshold
        value_tensor_filtered = value_tensor[mask]
        predictions_filtered = predictions[mask]

        targ = value_tensor_filtered.cpu().numpy().flatten()

        for cell_id in range(n_cell_ids):
            pred = predictions_filtered[:, pred_slice[0]:pred_slice[1], cell_id].sum(dim=1).cpu().numpy()
            corr, _ = pearsonr(np.log1p(pred), np.log1p(targ))
            corrs_df.loc[bed_cell, f'cell_{cell_id}'] = corr

            if verbose:
                print(f"{bed_cell} | cell_{cell_id}: r = {corr:.3f}")

    return corrs_df


def compute_ubiquitous_correlations(
    data_dir,
    predictions_path,
    cells,
    n_cell_ids=10,
    value_threshold=1,
    pred_slice=(447, 449),
    verbose=True,
):
   
    predictions = torch.load(predictions_path)

    # Initialize DataFrame
    corrs_df = pd.DataFrame(
        index=cells,
        columns=[f"cell_{i}" for i in range(n_cell_ids)],dtype=float
    )

    for cell in cells:
        # Load targets
        targets_df = pd.read_csv(
            f"{data_dir}/{cell}.bed",
            sep="\t",
            names=["chrom", "start", "end", "value", "fold"],
        )

        # Filter test fold
        value = targets_df.loc[targets_df["fold"] == "test", "value"]
        value_tensor = torch.tensor(value.values)

        # Apply value threshold
        mask = value_tensor > value_threshold
        value_tensor_filtered = value_tensor[mask]
        predictions_filtered = predictions[mask]

        targ = value_tensor_filtered.cpu().numpy().flatten()

        for cell_id in range(n_cell_ids):
            pred = (
                predictions_filtered[:, pred_slice[0]:pred_slice[1], cell_id]
                .cpu()
                .numpy()
                .sum(axis=1)
            )
            corr, _ = pearsonr(np.log1p(pred), np.log1p(targ))
            corrs_df.loc[cell, f"cell_{cell_id}"] = corr

            if verbose:
                print(f"{cell} - cell_{cell_id}: Pearson r = {corr:.3f}")

    return corrs_df


def compute_caai(pos_sad, neg_sad, model):

    """
    Compute the centered allelic imbalance (CAAI) from reference and
    alternative allele scores for positive and negative samples.

    """

    # Select slice indices based on model
    if model == 'enformer':
        index1, index2 = 447, 449
    elif model == 'borzoi_ensemble':
        index1, index2 = 0, 8
    elif model == 'borzoi_rep':
        index1, index2 = 0,8
    elif model == 'kidformer':
        index1, index2 = 1, 3
    elif model == 'chromekid':
        index1, index2 =0,1
    elif model == 'kidzoi':
        index1, index2 =4,12 
    elif model == 'alphagenome':
        index1, index2 =15 ,17

    if model =='sei':
        
        #positive samples
        pos_ref = pos_sad['REF'][:]
        pos_alt = pos_sad['ALT'][:]
        den = pos_ref + pos_alt
        
        pos_caai = np.divide(
        pos_ref,
        den,
        out=np.full_like(pos_ref, np.nan, dtype=float),
        where=den != 0)

        # Negative samples
        neg_ref = neg_sad['REF'][:]
        neg_alt = neg_sad['ALT'][:]
        den = neg_ref + neg_alt
        neg_caai = np.divide(
        neg_ref,
        den,
        out=np.full_like(neg_ref, np.nan, dtype=float),
        where=den != 0)
        sad_caai = np.concatenate([pos_caai, neg_caai])
        
        sad_caai = np.abs(sad_caai - 0.5)
        imb_labels = np.concatenate([
            np.ones(pos_ref.shape[0]),
            np.zeros(neg_ref.shape[0])
        ])

    else:
            
        pos_ref = pos_sad['REF'][:, index1:index2, :].sum(axis=1)
        pos_alt = pos_sad['ALT'][:, index1:index2, :].sum(axis=1)
        pos_denom = pos_ref + pos_alt
        pos_caai = np.divide(
            pos_ref,
            pos_denom,
            out=np.full_like(pos_ref, np.nan, dtype=float),
            where=pos_denom != 0
        )
        
        
        # Negative samples
        neg_ref = neg_sad['REF'][:, index1:index2, :].sum(axis=1)
        neg_alt = neg_sad['ALT'][:, index1:index2, :].sum(axis=1)
        neg_denom = neg_ref + neg_alt
        
        neg_caai = np.divide(
            neg_ref,
            neg_denom,
            out=np.full_like(neg_ref, np.nan, dtype=float),
            where=neg_denom != 0
        )
        
        sad_caai = np.concatenate([pos_caai, neg_caai])
        sad_caai = np.abs(sad_caai - 0.5)
        imb_labels = np.concatenate([
            np.ones(pos_ref.shape[0]),
            np.zeros(neg_ref.shape[0])
        ])
        
    targets = pos_sad['target_labels'][:]
    targets_df = pd.DataFrame({
    'index': np.arange(len(targets)),
    'description': targets
    })

    return sad_caai, imb_labels, targets_df


def sliding_caai_metrics(
    pos_sad,
    neg_sad,
    cell_id=9,
    step=200,
    values=None,
    start_min=16
):
    
    min_valid =10
    middle = pos_sad['REF'].shape[1] // 2
    # Determine window sizes
    if values is None:
        values = np.arange(start_min, middle, step)

    results = {"num": [], "auroc": [], "auprc": []}

    for num in values:
        if num == 0:

            start = middle - num
            end = middle + num
    
            # POS
            pos_ref = pos_sad['REF'][:, start:end+1, cell_id]
            pos_alt = pos_sad['ALT'][:, start:end+1, cell_id]
            pos_denom = pos_ref + pos_alt
            pos_caai = np.divide(
                pos_ref,
                pos_denom,
                out=np.full_like(pos_ref, np.nan, dtype=float),
                where=pos_denom != 0
            )

            # NEG
            neg_ref = neg_sad['REF'][:, start:end+1, cell_id]
            neg_alt = neg_sad['ALT'][:, start:end+1, cell_id]
            neg_denom = neg_ref + neg_alt
            neg_caai = np.divide(
                neg_ref,
                neg_denom,
                out=np.full_like(neg_ref, np.nan, dtype=float),
                where=neg_denom != 0
            )
    
            # Combine
            scores = np.concatenate([pos_caai, neg_caai])
            scores = np.abs(scores - 0.5)
    
            labels = np.concatenate([
                np.ones(pos_ref.shape[0]),
                np.zeros(neg_ref.shape[0])
            ])
    
            mask = np.isfinite(scores)
            scores, labels = scores[mask], labels[mask]
            if scores.size < min_valid or len(np.unique(labels)) < 2:
               continue
            # ROC
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = auc(fpr, tpr)
    
            # PR
            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = auc(recall, precision)
    
            results["num"].append(num)
            results["auroc"].append(auroc)
            results["auprc"].append(auprc)
        else:
            start = middle - num
            end = middle + num

            # POS
            
            pos_ref = pos_sad['REF'][:, start:end, cell_id].sum(axis=1)
            pos_alt = pos_sad['ALT'][:, start:end, cell_id].sum(axis=1)
            pos_denom = pos_ref + pos_alt
            pos_caai = np.divide(
                pos_ref,
                pos_denom,
                out=np.full_like(pos_ref, np.nan, dtype=float),
                where=pos_denom != 0
            )

            # NEG
            neg_ref = neg_sad['REF'][:, start:end, cell_id].sum(axis=1)
            neg_alt = neg_sad['ALT'][:, start:end, cell_id].sum(axis=1)
            neg_denom = neg_ref + neg_alt
            neg_caai = np.divide(
                neg_ref,
                neg_denom,
                out=np.full_like(neg_ref, np.nan, dtype=float),
                where=neg_denom != 0
            )
    
            # Combine
            scores = np.concatenate([pos_caai, neg_caai])
            scores = np.abs(scores - 0.5)
    
            labels = np.concatenate([
                np.ones(pos_ref.shape[0]),
                np.zeros(neg_ref.shape[0])
            ])
    
            mask = np.isfinite(scores)
            scores, labels = scores[mask], labels[mask]

            if scores.size < min_valid or len(np.unique(labels)) < 2:
               continue
    
            # ROC
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = auc(fpr, tpr)
    
            # PR
            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = auc(recall, precision)
    
            results["num"].append(num)
            results["auroc"].append(auroc)
            results["auprc"].append(auprc)

    return results



def compute_sad(pos_sad, neg_sad, model='kidzoi'):

    if model == 'enformer':
        bins = 8
    elif model == 'borzoiens':
        bins= 32
    elif model == 'borzoi':
        bins=32
    elif model == 'kidzoi':
        bins=32
    elif model == 'kidformer':
        bins=8
    elif model == 'alphagenome':
        bins=8
    
    if model=='sei':
        sad_pos = pos_sad['SAD']
        sad_neg = neg_sad['SAD'][:]
        sad = np.abs(np.concatenate([sad_pos, sad_neg]))
        labels = np.concatenate([np.ones(sad_pos.shape[0]), np.zeros(sad_neg.shape[0])])
        targets = pos_sad['target_labels']
        targets_df = pd.DataFrame({
            'index': np.arange(len(targets)),
            'description': targets
        })
    
    else:
        total_bins = pos_sad['REF'].shape[1]
        center = total_bins // 2
        index1 = center - (bins // 2)
        index2 = center + (bins//2)

        # Postives
        pos_ref = np.sum(pos_sad['REF'][:, index1:index2, :], axis=1)
        pos_alt = np.sum(pos_sad['ALT'][:, index1:index2, :], axis=1)
        sad_pos = pos_alt - pos_ref
    
        # Negatives
        neg_ref = np.sum(neg_sad['REF'][:, index1:index2, :], axis=1)
        neg_alt = np.sum(neg_sad['ALT'][:, index1:index2, :], axis=1)
        sad_neg = neg_alt - neg_ref
               
        sad = np.abs(np.concatenate([sad_pos, sad_neg]))
        labels = np.concatenate([np.ones(sad_pos.shape[0]), np.zeros(sad_neg.shape[0])])
       
        targets = pos_sad['target_labels']
        targets_df = pd.DataFrame({
            'index': np.arange(len(targets)),
            'description': targets
        })

    return sad, labels, targets_df



def sliding_sad_metrics(
    all_scores_pos,
    all_scores_neg,
    cell_id,
    values=None, 
    title="SAD Metrics Across Window Sizes"
):
    min_valid = 10
    seq_len = all_scores_pos['REF'].shape[1]
    middle = seq_len // 2

    if values is None:
        values = np.arange(32, seq_len // 2, 50)

    results = {"num": [], "auroc": [], "auprc": []}

    for num in values:

        if num == 0:
        
            start = middle
            end = middle

            pos_ref = all_scores_pos['REF'][:, start:end+1, cell_id]
            pos_alt = all_scores_pos['ALT'][:, start:end+1, cell_id]

            pos_ref_sum = pos_ref.sum(axis=1)
            pos_alt_sum = pos_alt.sum(axis=1)
            sad_pos = (pos_alt_sum - pos_ref_sum)

            # -----------------------
            # NEGATIVE SAMPLES
            # -----------------------
            neg_ref = all_scores_neg['REF'][:, start:end+1, cell_id]
            neg_alt = all_scores_neg['ALT'][:, start:end+1, cell_id]

            neg_ref_sum = neg_ref.sum(axis=1)
            neg_alt_sum = neg_alt.sum(axis=1)
            sad_neg = (neg_alt_sum - neg_ref_sum)

            # -----------------------
            # Combine POS and NEG
            # -----------------------
            labels = np.concatenate([np.ones(sad_pos.shape[0]), np.zeros(sad_neg.shape[0])])
            scores = np.concatenate([sad_pos, sad_neg])
            scores = np.abs(scores)

            valid = np.isfinite(scores)
            scores = scores[valid]
            labels = labels[valid]

            if scores.size < min_valid or len(np.unique(labels)) < 2:
                continue

            # -----------------------
            # Metrics
            # -----------------------
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = auc(fpr, tpr)
            #auprc = average_precision_score(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = auc(recall, precision)

            # Store results
            results["num"].append(num)
            results["auroc"].append(auroc)
            results["auprc"].append(auprc)

        else:
            start = middle - num
            end = middle + num

            pos_ref = all_scores_pos['REF'][:, start:end, cell_id]
            pos_alt = all_scores_pos['ALT'][:, start:end, cell_id]

            pos_ref_sum = pos_ref.sum(axis=1)
            pos_alt_sum = pos_alt.sum(axis=1)
            sad_pos = (pos_alt_sum - pos_ref_sum)

            # -----------------------
            # NEGATIVE SAMPLES
            # -----------------------
            neg_ref = all_scores_neg['REF'][:, start:end, cell_id]
            neg_alt = all_scores_neg['ALT'][:, start:end, cell_id]

            neg_ref_sum = neg_ref.sum(axis=1)
            neg_alt_sum = neg_alt.sum(axis=1)
            sad_neg = (neg_alt_sum - neg_ref_sum)

            # -----------------------
            # Combine POS and NEG
            # -----------------------
            labels = np.concatenate([np.ones(sad_pos.shape[0]), np.zeros(sad_neg.shape[0])])
            scores = np.concatenate([sad_pos, sad_neg])
            scores = np.abs(scores)

            valid = np.isfinite(scores)
            scores = scores[valid]
            labels = labels[valid]
            
            if scores.size < min_valid or len(np.unique(labels)) < 2:
                continue

            # -----------------------
            # Metrics
            # -----------------------
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = auc(fpr, tpr)
            #auprc = average_precision_score(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = auc(recall, precision)

            # Store results
            results["num"].append(num)
            results["auroc"].append(auroc)
            results["auprc"].append(auprc)

   

    return results






def compute_and_plot_metrics(sad_caai, labels, targets, save_path=None):

    min_valid = 10
    target_idx = np.array(targets['index'])
    metrics_data = []

    for ti in target_idx:
        imb_preds = sad_caai[:, ti]

        valid = np.isfinite(imb_preds)
        imb_preds = imb_preds[valid]
        imb_lbls_clean = labels[valid]
       
        if imb_preds.size < min_valid or len(np.unique(imb_lbls_clean)) < 2:
                continue
                
        # Compute AUROC
        fpr, tpr, _ = roc_curve(imb_lbls_clean, imb_preds)
        roc_auc = auc(fpr, tpr)

       
        precision, recall, _ = precision_recall_curve(imb_lbls_clean, imb_preds)
        auprc = auc(recall, precision)
        
        legend_label = targets.loc[targets['index'] == ti, 'description'].values[0]
        

        metrics_data.append({
            'Target Index': ti,
            'Identifier': legend_label,
            'AUROC': roc_auc,
            'AUPRC': auprc
        })

    metrics_df = pd.DataFrame(metrics_data)

    if save_path:
        metrics_df.to_csv(save_path, sep='\t', index=False)

    return metrics_df




cell_info = [
    {"id": 0, "description": "Imm"},
    {"id": 1, "description": "Str"},
    {"id": 2, "description": "Pod"},
    {"id": 3, "description": "CD"},
    {"id": 4, "description": "PE"},
    {"id": 5, "description": "Tcell"},
    {"id": 6, "description": "End"},
    {"id": 7, "description": "DT"},
    {"id": 8, "description": "PT"},
    {"id": 9, "description": "LOH"},
]

def plot_roc_pr_cells(labels_all, sad_all, cell_info, suptitle=None, save_pdf=False, filename='test'):
   
    colors = plt.cm.tab10.colors 
    min_valid = 10
    # Define color mapping for each cell type
    cell_colors = {
        'Imm': colors[0],
        'Str': colors[1],
        'Pod': colors[2],
        'CD': colors[3],
        'PE': colors[4],
        'Tcell': colors[5],
        'End': colors[6], 
        'DT': colors[7],
        'PT': colors[8],
        'LOH': colors[9]
    }
    
    metrics = []
    for cell in cell_info:
        scores = sad_all[:, cell['id']]
        mask = np.isfinite(scores)
        scores, labels = scores[mask], labels_all[mask]

        if scores.size < min_valid or len(np.unique(labels)) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = auc(recall, precision)
        metrics.append({
            "cell": cell,
            "roc_auc": roc_auc,
            "auprc": auprc,
            "fpr": fpr,
            "tpr": tpr,
            "recall": recall,
            "precision": precision
        })
    
    # Sort metrics by AUROC and AUPRC separately
    metrics_roc_sorted = sorted(metrics, key=lambda x: x['roc_auc'], reverse=True)
    metrics_pr_sorted = sorted(metrics, key=lambda x: x['auprc'], reverse=True)
    
    plt.figure(figsize=(6, 2.5))
    
    # ROC subplot
    plt.subplot(1, 2, 1)
    for m in metrics_roc_sorted:
        cell_desc = m['cell']['description']
        color = cell_colors[cell_desc]  # Use the fixed color for this cell type
        plt.plot(m['fpr'], m['tpr'], color=color, 
                label=f"{cell_desc} ({m['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', alpha=0.3, linestyle='--', linewidth=0.8)
    plt.xlabel('False Positive Rate', fontsize=6)
    plt.ylabel('True Positive Rate', fontsize=6)
    plt.grid(False)
    plt.legend(loc='lower right', frameon=False, fontsize=6)
    
    ax1 = plt.gca()
    ax1.tick_params(axis='both',
                which='major',
                direction='out',
                bottom=True, top=False,
                left=True, right=False,
                width=0.5,
                colors='black',
                labelsize=6)
    
    for spine in ax1.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('gray')
    
    # PR subplot
    plt.subplot(1, 2, 2)
    for m in metrics_pr_sorted:
        cell_desc = m['cell']['description']
        color = cell_colors[cell_desc]  # Use the fixed color for this cell type
        plt.plot(m['recall'], m['precision'], color=color, 
                label=f"{cell_desc} ({m['auprc']:.2f})")
    plt.xlabel('Recall', fontsize=6)
    plt.ylabel('Precision', fontsize=6)
    plt.grid(False)
    plt.legend(loc='upper right', frameon=False, fontsize=6)
    ax2 = plt.gca()
    ax2.tick_params(axis='both',
                which='major',
                direction='out',
                bottom=True, top=False,
                left=True, right=False,
                width=0.5,
                colors='black',
                labelsize=6)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('gray')
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=6)
    
    plt.tight_layout()
    
    if save_pdf:
        plt.savefig(f'{filename}.pdf', bbox_inches='tight')
        print(f"Saved figure as {filename}")
    
    plt.show()





def plot_boxplot(data, metric_name, top1_values, model_order, palette, colors, filename, yticks=None):
    """Plot boxplot with horizontal reference lines"""
    plt.figure(figsize=(3.2, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.boxplot(
            x="Model", y=metric_name,
            data=data,
            order=model_order,
            palette=palette,
            flierprops=dict(markersize=3) 
        )
    
    # Add horizontal reference lines sorted by value
    for model, value in sorted(top1_values.items(), key=lambda x: x[1], reverse=True):
        plt.axhline(
            y=value, color=colors.get(model, 'gray'),
            linestyle='--', linewidth=1, alpha=1.0,
            label=f'{model}: {value:.2f}'
        )
    
    plt.ylabel(metric_name, fontsize=6)
    plt.xlabel("")
    plt.xticks(rotation=45)
    
    ax = plt.gca()
    if yticks is not None:
        ax.set_yticks(yticks)
    
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0,
        frameon=False,
        fontsize=6
    )

    for patch in ax.patches:
        patch.set_alpha(1.0)
    
    ax.tick_params(
    axis='both',
    which='major',
    labelsize=6,
    width=0.5,
    color='black',
    direction='out',  # Ticks point outward
    bottom=True,  # Show bottom ticks
    left=True,  # Show left ticks
    top=False,
    right=False,
) 
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color('gray')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()