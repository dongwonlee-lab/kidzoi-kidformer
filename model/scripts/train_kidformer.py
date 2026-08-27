import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from enformer_pytorch.finetune import HeadAdapterWrapper
import gc
import argparse
import pandas as pd
import math 
import random
from pyfaidx import Fasta
import pyBigWig

from tqdm import tqdm
from enformer_pytorch import from_pretrained
from enformer_pytorch import Enformer

import wandb
import inspect
from utils import *
seed = 42 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)          
torch.cuda.manual_seed(seed)     
torch.cuda.manual_seed_all(seed)
torch.set_float32_matmul_precision('high')

num_epochs = 4
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 1000




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--targets_file', default="/path/to/targets_file.txt", help='path to train data')
    parser.add_argument('--genome', default="/path/to/genome.fa", help='Genome FASTA')
    parser.add_argument('--bed_file', default="path/to/sequences_human_enformer.bed", help='Targets file')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f'Using device: {device}')
    args = parser.parse_args()
    genome = args.genome
    sequences = args.bed_file
    targets_file = args.targets_file
    model_name = "kidformer"
    print("Loading training data...")

     # Create data loaders
    batch_size = 1
    train_loader = make_loader(sequences, targets_file, genome,
                      split='train', seq_len=196608, target_len=114688,
                      batch_size=batch_size, num_workers=batch_size, shuffle=True, rc_aug=True, shift_aug=True)

    val_loader = make_loader(sequences, targets_file, genome,
                      split='valid', seq_len=196608, target_len=114688,
                      batch_size=batch_size, num_workers=batch_size, shuffle=False, rc_aug=False, shift_aug=False) 

    print(f"Total training batches: {len(train_loader)} with batch size {batch_size}")

    # Gradient accumulation settings
    desired_batch_size = 1
    accumulation_steps = desired_batch_size // batch_size 
    print(f"Using gradient accumulation with {accumulation_steps} steps (effective batch size: {desired_batch_size})")
    
    # Training loop
    
    max_steps = ((len(train_loader))// accumulation_steps) * num_epochs
    total_steps = len(train_loader) // accumulation_steps * num_epochs
       
    print("Loading pretrained Enformer model...")
    num_tracks=10
   
    enformer = from_pretrained('EleutherAI/enformer-official-rough')
    
    # Create the HeadAdapterWrapper for fine-tuning with 10 output tracks
    print("Setting up HeadAdapterWrapper for 10 output tracks...")
    model = HeadAdapterWrapper(
        enformer=enformer,
        num_tracks=num_tracks, 
        post_transformer_embed=False  # Use embeddings right after transformer block
    ).to(device)
    
    # Set model to training mode
    model.train()
    model = torch.compile(model)
    wandb.init(project=model_name)
    loss_accum = 0.0
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=max_lr, device=device)
    print(f"Starting training for {num_epochs} epochs...")
    step = 0 
    stepi = []
    lossi = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []  # Store loss per optimization step, not per batch
        
        # Training
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            
            # Zero gradients at start of accumulation step
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
                loss_accum = 0.0
            
            seq = batch[0].permute(0,2,1).to(device)
            target = batch[1].to(device)
             
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # Forward pass
                loss = model(seq, target=target)

            # Scale loss for gradient accumulation 
            loss = loss / accumulation_steps
            loss_accum += loss.detach()
            #print(loss_accum)      
            # Backward pass
            loss.backward()
            
            # Gradient accumulation - update every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
                
                # Update learning rate per step (not per epoch)
                lr = get_lr(step, warmup_steps, max_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Optimizer step
                optimizer.step()
                
                # Store loss for this optimization step (unscaled)
                epoch_losses.append(loss_accum.item())
                print(f"Step {step}, Loss: {loss_accum.item():.4f}, LR: {lr:.4e}, Norm: {norm:.4f}")
                step += 1  # Increment global step counter
                stepi.append(step)
                lossi.append(loss_accum.item())
                #wandb.log({"train_loss": loss_accum.item(), "step": step})
               
                wandb.log({"train_loss": loss_accum.item(), "norm": norm, "lr": lr}, step=step)
                
        #Handle remaining gradients at end of epoch
        if len(train_loader) % accumulation_steps != 0:
            # Gradient clipping
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update learning rate
            lr =get_lr(step, warmup_steps, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.step()
            
            # Add the final partial step loss
            epoch_losses.append(loss_accum.item())
            step += 1
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                seq = batch[0].permute(0,2,1).to(device)
                target = batch[1].to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    loss = model(seq, target=target)
                val_losses.append(loss.item())
            
                    
        avg_train_loss = np.mean(epoch_losses)
        avg_val_loss = np.mean(val_losses)
        # At the end of each epoch, after computing avg_val_loss:
        
        wandb.log({
        "epoch_train_loss": avg_train_loss,
        "epoch_val_loss": avg_val_loss,
        "epoch": epoch + 1
        }, step=step)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Steps completed: {step}")
        print("-" * 50)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Save both the model weights AND useful metadata
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'step': step,
            }
            torch.save(checkpoint, f"{model_name}_best.pth")
            print(f"  New best model saved! Val loss: {best_val_loss:.4f}")
        else:
            print(f"  (No improvement. Best val loss so far: {best_val_loss:.4f})")
    
    print("Saving fine-tuned model...")
    torch.save(model.state_dict(), f"{model_name}.pth")
           

if __name__ == "__main__":
    main()
