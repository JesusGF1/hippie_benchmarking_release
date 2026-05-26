import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


def extend_sequence_interpolation(tensor, target_size):
    """
    Extend a sequence to target_size by interpolating additional points within the existing boundaries.
    If the sequence is already longer than target_size, it will be downsampled.
    
    Args:
        tensor: Input tensor of shape (batch, channels, length) or (batch, length)  
        target_size: Desired length
        
    Returns:
        Extended/resampled tensor with target_size length
    """
    if tensor.dim() == 2:
        # Add channel dimension if missing
        tensor = tensor.unsqueeze(1)
    
    current_size = tensor.size(-1)
    
    if current_size == target_size:
        return tensor
    else:
        # For both extension and downsampling, F.interpolate works well
        # The key is that it maintains the signal boundaries and interpolates within them
        return F.interpolate(tensor, size=(target_size,), mode="linear", align_corners=False)


def none_safe_collate(data):
    samples = [x[0] for x in data if x is not None]
    labels = [x[1] for x in data if x is not None]
    
    data_results = {key: [] for key in samples[0].keys()}
    for sample in samples:
        for key, value in sample.items():
            data_results[key].append(value)
    samples = {key: torch.stack(value) for key, value in data_results.items()}

    if labels[0].ndim == 0:
        labels = torch.as_tensor(labels).long()
    else:
        labels = torch.stack(labels).long()

    return samples, labels

class MultiModalEphysDataset(Dataset):
    """Dataset for multiple modalities with consistent preprocessing."""
    def __init__(self, data_dict, labels, mode="multi", normalize=True, modality_sizes=None):
        """
        Initialize a multi-modal ephys dataset.
        
        Args:
            data_dict (dict): Dictionary mapping modality names to data arrays
            labels (np.ndarray): Array of labels
            mode (str): Either one of the modality keys or "multi"
            normalize (bool): Whether to perform normalization
            modality_sizes (dict, optional): Target sizes for each modality
        """
        self.modalities = {}
        for mod_name, data in data_dict.items():
            self.modalities[mod_name] = np.array(data)
        
        self.labels = np.array(labels)
        
        if mode != "multi" and mode not in self.modalities:
            print(f"Mode '{mode}' must be 'multi' or one of the modality keys: {list(self.modalities.keys())}")
        self.mode = mode
        
        n_samples = len(self.labels)
        for mod_name, data in self.modalities.items():
            assert len(data) == n_samples, f"Modality {mod_name} has {len(data)} samples but expected {n_samples}"
        
        default_sizes = {
            "wave": 50,
            "isi": 100,
            "acg": 100,
        }
        
        self.modality_sizes = default_sizes.copy()
        if modality_sizes:
            self.modality_sizes.update(modality_sizes)
        
        self.normalize = normalize
    
    def process_modality(self, mod_name, data, idx):
        """Process a single modality sample."""
        tensor = torch.as_tensor(data[idx, ...]).float()
        
        # Check for NaN or infinite values in raw data
        # if torch.isnan(tensor).any():
        #     print(f"NaN values detected in raw data for sample {idx}, modality '{mod_name}'")
            
        # if torch.isinf(tensor).any():
        #     print(f"Infinite values detected in raw data for sample {idx}, modality '{mod_name}'")
            
        # Apply log transform for ISI
        if mod_name == "isi":
            tensor = torch.log(tensor + 1)
            
        # Check after log transform
        # if torch.isnan(tensor).any():
        #     print(f"NaN values detected after log transform for sample {idx}, modality '{mod_name}'")
            
        # Apply normalization
        if self.normalize:
            min_val = tensor.min()
            max_val = tensor.max()
            
            tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
            tensor = tensor * 2 - 1
            
            # if torch.isnan(tensor).any():
            #     print(f"NaN values detected after normalization for sample {idx}, modality '{mod_name}'")
            
        tensor = tensor.view(1, 1, -1)
        
        target_size = self.modality_sizes.get(mod_name)
        if target_size:
            tensor = extend_sequence_interpolation(tensor, target_size)
            
            # if torch.isnan(tensor).any():
            #     print(f"NaN values detected after interpolation for sample {idx}, modality '{mod_name}'")
            
        tensor = tensor.view(1, -1)
        return tensor

    def __getitem__(self, idx):
        # Convert labels via Python int / list rather than numpy arrays so
        # that torch.tensor works consistently across numpy 1.x and 2.x.
        # (The torch 2.0–2.2 + numpy 2.x ABI combination can fail to read
        # numpy.int64 directly through torch.as_tensor.)
        raw = self.labels[idx]
        if np.ndim(raw) == 0:
            label = torch.tensor(int(raw), dtype=torch.long)
        else:
            label = torch.tensor([int(v) for v in np.asarray(raw).reshape(-1)], dtype=torch.long)
        
        if self.mode == "multi":
            result = {}
            for mod_name, data in self.modalities.items():
                result[mod_name] = self.process_modality(mod_name, data, idx)
                
                if torch.isnan(result[mod_name]).any() or torch.isinf(result[mod_name]).any():
                    print(f"NaN values detected in sample {idx}, modality '{mod_name}'")
                    return None
            return result, label
        else:
            tensor = self.process_modality(self.mode, self.modalities[self.mode], idx)
            
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"NaN values detected in sample {idx}, modality '{self.mode}'")
                return None 
            return tensor, label
        
    def __len__(self):
        return len(self.labels)


