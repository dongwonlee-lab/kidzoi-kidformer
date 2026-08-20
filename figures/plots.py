
def compute_cell_specific_correlations(
    target_dir,
    pred_dir,
    cells_bed,
    cells_pred,
    
    n_cell_ids=10,
    value_threshold=1,
    pred_slice=(447, 449),
    verbose=True,
    model='enformer'
):
   

    corrs_df = pd.DataFrame(
        index=cells_bed,
        columns=[f'cell_{i}' for i in range(n_cell_ids)],
        dtype=float
    )

    for bed_cell, pred_cell in zip(cells_bed, cells_pred):
        # Load targets
        targets_df = pd.read_csv(
            f"{target_dir}/{bed_cell}.bed",
            sep="\t",
            names=['chrom', 'start', 'end', 'value', 'fold']
        )

        # Load predictions
        if model =='borzoi':
            predictions = torch.load(f"{pred_dir}/{pred_cell}.pt").permute(0, 2, 1)
        else:
            predictions = torch.load(f"{pred_dir}/{pred_cell}.pt")

        # Filter test fold and apply threshold
        value = targets_df.loc[targets_df['fold'] == 'test', 'value']
        value_tensor = torch.tensor(value.values, dtype=torch.float32)
        mask = value_tensor > value_threshold
        value_tensor_filtered = value_tensor[mask]
        predictions_filtered = predictions[mask]

        targ = value_tensor_filtered.cpu().numpy().flatten()

        for cell_id in range(n_cell_ids):
            pred = predictions_filtered[:, pred_slice[0]:pred_slice[1], cell_id].mean(dim=1).cpu().numpy()
            corr, _ = pearsonr(np.log1p(pred), np.log1p(targ))
            corrs_df.loc[bed_cell, f'cell_{cell_id}'] = corr

            if verbose:
                print(f"{bed_cell} | cell_{cell_id}: r = {corr:.3f}")

    return corrs_df


def compute_cellwise_correlations_ubiquitous(
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
        columns=[f"cell_{i}" for i in range(n_cell_ids)],
        dtype=float,
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
        value_tensor = torch.tensor(value.values, dtype=torch.float32)

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

            corr, _ = pearsonr(np.log(pred + 1), np.log(targ + 1))
            corrs_df.loc[cell, f"cell_{cell_id}"] = corr

            if verbose:
                print(f"{cell} - cell_{cell_id}: Pearson r = {corr:.3f}")

    return corrs_df


def sliding_sad_metrics(
    pos_sad,
    neg_sad,
    cell_id=9,
    step=200,
    seqlen=16384,
    values=None,
    start_min=4,
    eps=1e-8,
    min_valid=10,
):
    middle = seqlen // 2
    max_window = pos_sad['REF'].shape[1] // 2

    # Determine window sizes
    if values is None:
        values = np.arange(start_min, max_window, step)

    results = {"num": [], "auroc": [], "auprc": []}

    for num in values:
        if num == 0:

            start = middle - num
            end = middle + num
    
            # POS
            pos_ref = pos_sad['REF'][:, start:end+1, cell_id]#.mean(axis=1)
            pos_alt = pos_sad['ALT'][:, start:end+1, cell_id]#.mean(axis=1)
            pos_caai = pos_ref / (pos_ref + pos_alt + eps)
    
            # NEG
            neg_ref = neg_sad['REF'][:, start:end+1, cell_id]#.mean(axis=1)
            neg_alt = neg_sad['ALT'][:, start:end+1, cell_id]#.mean(axis=1)
            neg_caai = neg_ref / (neg_ref + neg_alt + eps)
    
            # Combine
            sad_caai = np.concatenate([pos_caai, neg_caai])
            sad_caai = np.abs(sad_caai - 0.5)
    
            labels = np.concatenate([
                np.ones(pos_ref.shape[0]),
                np.zeros(neg_ref.shape[0])
            ])
    
            #scores = sad_caai[:, cell_id]
            scores = sad_caai
            # ROC
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = auc(fpr, tpr)
    
            # PR
            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = average_precision_score(labels, scores)
    
            results["num"].append(num)
            results["auroc"].append(auroc)
            results["auprc"].append(auprc)
        else:
            start = middle - num
            end = middle + num

            # POS
            pos_ref = pos_sad['REF'][:, start:end, cell_id].mean(axis=1)
            pos_alt = pos_sad['ALT'][:, start:end, cell_id].mean(axis=1)
            pos_caai = pos_ref / (pos_ref + pos_alt + eps)
    
            # NEG
            neg_ref = neg_sad['REF'][:, start:end, cell_id].mean(axis=1)
            neg_alt = neg_sad['ALT'][:, start:end, cell_id].mean(axis=1)
            neg_caai = neg_ref / (neg_ref + neg_alt + eps)
    
            # Combine
            sad_caai = np.concatenate([pos_caai, neg_caai])
            sad_caai = np.abs(sad_caai - 0.5)
    
            labels = np.concatenate([
                np.ones(pos_ref.shape[0]),
                np.zeros(neg_ref.shape[0])
            ])
    
            #scores = sad_caai[:, cell_id]
            scores = sad_caai
    
            
            valid = np.isfinite(scores)
    
            scores = scores[valid]
            labels = labels[valid]
    
            #Skip if too few points remain
            if scores.size < min_valid or len(np.unique(labels)) < 2:
               continue
    
        
    
            # ROC
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = auc(fpr, tpr)
    
            # PR
            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = average_precision_score(labels, scores)
    
            # Store
            results["num"].append(num)
            results["auroc"].append(auroc)
            results["auprc"].append(auprc)

    return results


def clip_float(x, dtype=np.float16):
    return np.clip(x, np.finfo(dtype).min, np.finfo(dtype).max)

    
def compute_sad_metrics(
    all_scores_pos,
    all_scores_neg,
    pos_index,
    cell_id,
    values=None, 
    title="SAD Metrics Across Window Sizes"
):
   
    seq_len = all_scores_pos['REF'].shape[1]
    middle = seq_len // 2

    if values is None:
        values = np.arange(32, seq_len // 2, 50)

    results = {"num": [], "auroc": [], "auprc": []}

    for num in values:

        if num == 0:
            continue
        
        start = middle - num
        end = middle + num

        pos_ref = all_scores_pos['REF'][:, start:end, cell_id].astype(np.float32)
        pos_alt = all_scores_pos['ALT'][:, start:end, cell_id].astype(np.float32)

        pos_ref_sum = pos_ref.sum(axis=1)
        pos_alt_sum = pos_alt.sum(axis=1)
        sad_pos = clip_float(pos_alt_sum - pos_ref_sum)
        sad_pos = sad_pos[pos_index]

        # -----------------------
        # NEGATIVE SAMPLES
        # -----------------------
        neg_ref = all_scores_neg['REF'][:, start:end, cell_id].astype(np.float32)
        neg_alt = all_scores_neg['ALT'][:, start:end, cell_id].astype(np.float32)

        neg_ref_sum = neg_ref.sum(axis=1)
        neg_alt_sum = neg_alt.sum(axis=1)
        sad_neg = clip_float(neg_alt_sum - neg_ref_sum)

        # -----------------------
        # Combine POS and NEG
        # -----------------------
        labels = np.concatenate([np.ones(sad_pos.shape[0]), np.zeros(sad_neg.shape[0])])
        sad_all = np.concatenate([sad_pos, sad_neg])
        sad_all = np.abs(sad_all)

        #scores = sad_all[:, cell_id]
        scores = sad_all

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


def compute_and_plot_metrics(sad_caai, imb_labels, targets, save_path=None):
    
    target_idx = np.array(targets['index'])
    metrics_data = []

    for ti in target_idx:
        imb_preds = sad_caai[:, ti]

       
        valid_mask = ~np.isnan(imb_preds) & ~np.isnan(imb_labels)
        imb_preds = imb_preds[valid_mask]
        imb_lbls_clean = imb_labels[valid_mask]

       
        if len(imb_lbls_clean) == 0 or np.unique(imb_lbls_clean).size < 2:
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

    # Save if requested
    if save_path:
        metrics_df.to_csv(save_path, sep='\t', index=False)

    return metrics_df


def compute_caai(pos_sad, neg_sad, model):

    """
    Compute the centered allelic imbalance (CAAI) from reference and
    alternative allele scores for positive and negative samples.

    Allelic imbalance is then computed as
    REF / (REF + ALT) and centered around 0.5 by taking the absolute
    difference |CAAI - 0.5|. Values closer to 0 indicate balanced REF/ALT
    activity, whereas larger values indicate greater allelic imbalance.

    For the Sei model, REF and ALT scores are used directly without averaging
    across a target/track dimension.

    Parameters
    ----------
    pos_sad : dict-like
        SAD results for positive samples. Must contain 'REF', 'ALT', and
        'target_labels'.
    neg_sad : dict-like
        SAD results for negative samples. Must contain 'REF' and 'ALT'.
    model : str
        Model used to determine the target/track range for aggregation.
        Supported models are 'enformer', 'borzoi_ensemble', 'borzoi_rep',
        'kidformer', 'chromkid', 'kidzoi', 'alphagenome', and 'sei'.

    Returns
    -------
    sad_caai : numpy.ndarray
        Absolute centered allelic imbalance values for the combined positive
        and negative samples.
    imb_labels : numpy.ndarray
        Binary labels indicating sample class: 1 for positive samples and
        0 for negative samples.
    targets_df : pandas.DataFrame
        DataFrame containing the target indices and corresponding target
        descriptions from the positive samples.
    """

    eps = 1e-8
    # Select slice indices based on model
    if model == 'enformer':
        index1, index2 = 447, 449
    elif model == 'borzoi_ensemble':
        index1, index2 = 0, 8
    elif model == 'borzoi_rep':
        index1, index2 = 0,8
    elif model == 'kidformer':
        index1, index2 = 1, 3
    elif model == 'chromkid':
        index1, index2 =0,1
    elif model == 'kidzoi':
        index1, index2 =4,12 

    elif model == 'alphagenome':
        index1, index2 =12 ,20 #12 ,14

    if model =='sei':
        
        #positive samples
        pos_ref = pos_sad['REF'][:]
        pos_alt = pos_sad['ALT'][:]
        den = pos_ref + pos_alt
        pos_caai = np.divide(
        neg_ref,
        den,
        out=np.zeros_like(neg_ref, dtype=float),
        where=den != 0)
        # Negative samples
        neg_ref = neg_sad['REF'][:]
        neg_alt = neg_sad['ALT'][:]
        den = neg_ref + neg_alt
        neg_caai = np.divide(
        neg_ref,
        den,
        out=np.zeros_like(neg_ref, dtype=float),
        where=den != 0)
        sad_caai = np.concatenate([pos_caai, neg_caai])
        
        sad_caai = np.abs(sad_caai - 0.5)
        imb_labels = np.concatenate([
            np.ones(pos_ref.shape[0]),
            np.zeros(neg_ref.shape[0])
        ])

    else:
            
       
        pos_ref = pos_sad['REF'][:, index1:index2, :].mean(axis=1)
        pos_alt = pos_sad['ALT'][:, index1:index2, :].mean(axis=1)
        pos_denom = pos_ref + pos_alt
        pos_caai = np.divide(
            pos_ref,
            pos_denom,
            out=np.full_like(pos_ref, np.nan, dtype=float),
            where=pos_denom != 0
        )
        
        
        # Negative samples
        neg_ref = neg_sad['REF'][:, index1:index2, :].mean(axis=1)
        neg_alt = neg_sad['ALT'][:, index1:index2, :].mean(axis=1)
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
    
    # Define color mapping for each cell type
    cell_colors = {
        'Imm': colors[0],
        'Str': colors[1],
        'Pod': colors[2],
        'CD': colors[3],
        'PE': colors[4],
        'Tcell': colors[5],
        'End': colors[6],  # Changed from 'END' to 'End' to match your cell_info
        'DT': colors[7],
        'PT': colors[8],
        'LOH': colors[9]
    }
    
    metrics = []
    for cell in cell_info:
        scores = sad_all[:, cell['id']]
        fpr, tpr, _ = roc_curve(labels_all, scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(labels_all, scores)
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
    #plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, 
      #     fontsize=6, borderaxespad=0)
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
