import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def get_token_type_mask(input_ids, labels, processor, max_seq_len=None):
    """
    Classifies each token in the sequence into one of 4 categories:
    0: Padding
    1: Vision (Image/Video placeholders)
    2: User Text (Ignored)
    3: Assistant Text (Trainable)
    
    Now handles manual padding to max_seq_len for visualization.
    """
    tokenizer = processor.tokenizer
    
    # --- 1. Identify Special IDs ---
    pad_id = tokenizer.pad_token_id
    
    # Qwen2.5-VL specific vision tokens
    try:
        image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    except:
        image_pad_id = 151655 
        video_pad_id = 151656

    # Convert to numpy
    if isinstance(input_ids, torch.Tensor):
        ids_np = input_ids.detach().cpu().numpy()
    else:
        ids_np = np.array(input_ids)

    if isinstance(labels, torch.Tensor):
        lbls_np = labels.detach().cpu().numpy()
    else:
        lbls_np = np.array(labels)
    
    # --- 2. Initial Classification (Variable Length) ---
    # Default to User Text (2)
    current_len = len(ids_np)
    token_types = np.full(current_len, 2, dtype=np.int32) 

    # Assistant: Label is NOT -100
    token_types[lbls_np != -100] = 3
    
    # Vision: Input ID matches image/video pad tokens
    is_vision = (ids_np == image_pad_id) | (ids_np == video_pad_id)
    token_types[is_vision] = 1
    
    # Padding (in existing data): Input ID matches pad_token_id
    is_padding = (ids_np == pad_id)
    token_types[is_padding] = 0
    
    # --- 3. Manual Padding to max_seq_len ---
    if max_seq_len is not None and current_len < max_seq_len:
        # Create full buffer initialized to 0 (Padding)
        padded_mask = np.zeros(max_seq_len, dtype=np.int32)
        # Fill the start with actual data
        padded_mask[:current_len] = token_types
        return padded_mask
    
    return token_types

def plot_batches(batches, processor, save_path="batch_visualization.png", max_width_display=4096):
    """
    Generates a stacked bar plot for a list of batches.
    Overlays red lines based on 'attention_mask' (cu_seqlens) to verify packing boundaries.
    """
    num_batches = len(batches)
    if num_batches == 0:
        print("No batches to plot.")
        return

    # Prepare figure
    fig, ax = plt.subplots(figsize=(20, num_batches * 0.8)) 
    
    # Color Map: Pad(Grey), Vision(Blue), User(Orange), Assistant(Green)
    colors = ['#E0E0E0', '#4A90E2', '#F5A623', '#7ED321'] 
    labels_map = {0: 'Padding', 1: 'Vision Tokens', 2: 'User Prompt', 3: 'Assistant (Trainable)'}
    
    print(f"Generating visualization for {num_batches} batches...")

    for i, batch in enumerate(batches):
        input_ids = batch["input_ids"][0] 
        labels = batch["labels"][0]
        
        # --- 1. Get Mask (Padded to max_width_display) ---
        # We pass max_width_display here to ensure the bar fills the plot width
        token_types = get_token_type_mask(input_ids, labels, processor, max_seq_len=max_width_display)
        
        # Create an image row for this batch
        bar_height = 10
        batch_img = np.tile(token_types, (bar_height, 1))
        
        # Extent: [x_min, x_max, y_min, y_max]
        y_bottom = i
        y_top = i + 0.8
        # Use actual len(token_types) which is now max_width_display
        extent = [0, len(token_types), y_bottom, y_top]
        
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        ax.imshow(batch_img, cmap=cmap, aspect='auto', extent=extent, vmin=0, vmax=3, interpolation='nearest')

        # --- 2. Overlay Sample Boundaries (Red Lines) ---
        cu_seqlens = batch.get("attention_mask")
        
        if cu_seqlens is not None:
            if isinstance(cu_seqlens, torch.Tensor):
                boundaries = cu_seqlens.detach().cpu().numpy()
            else:
                boundaries = np.array(cu_seqlens)
            
            for b_idx in boundaries:
                # Only draw lines that fall within the visible area
                if b_idx > 0 and b_idx < len(token_types):
                    ax.vlines(x=b_idx, ymin=y_bottom, ymax=y_top, 
                              colors='red', linestyles='--', linewidth=2, alpha=0.9)

        # --- 3. Metrics Text ---
        non_pad = np.sum(token_types != 0)
        total = len(token_types) # Should be max_width_display now
        eff = (non_pad / total) * 100
        
        num_samples = len(boundaries) - 2 if cu_seqlens is not None else 1
        
        info_text = f"Eff: {eff:.1f}% | Samples: {num_samples}"
        ax.text(total + 50, i + 0.4, info_text, 
                va='center', fontsize=10, fontweight='bold', color='#333333')

    # Formatting
    ax.set_ylim(-0.5, num_batches)
    ax.set_xlim(0, max_width_display)
    ax.set_yticks(np.arange(num_batches) + 0.4)
    ax.set_yticklabels([f"Batch {i}" for i in range(num_batches)])
    ax.set_xlabel("Token Position")
    ax.set_title("Packed Batch Visualization (Red Lines = Sample Boundaries)")
    
    # Legend 
    patches = [mpatches.Patch(color=colors[i], label=labels_map[i]) for i in range(4)]
    from matplotlib.lines import Line2D
    patches.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Sample Start (cu_seqlens)'))
    
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05 if num_batches < 10 else -0.02), ncol=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.close()