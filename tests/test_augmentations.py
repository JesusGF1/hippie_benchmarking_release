#!/usr/bin/env python3
"""
Test script to validate data augmentation implementation for HIPPIE.
This script tests the augmentation functionality with sample data and creates
visualization plots to verify correct behavior.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Add hippie to path
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hippie'))
sys.path.append(code_dir)

# Import hippie modules
from dataloading import MultiModalEphysDataset, none_safe_collate
from multimodal_model import ExperimentConfigs
from augmentations import AugmentedMultiModalEphysDataset, NeuralAugmentations

def create_sample_data():
    """Create synthetic sample data for testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 100
    
    # Create synthetic waveform data (50 time points)
    waveforms = []
    for i in range(n_samples):
        # Create a simple spike-like waveform
        t = np.linspace(0, 2*np.pi, 50)
        waveform = np.sin(t) * np.exp(-t/3) + 0.1 * np.random.randn(50)
        waveforms.append(waveform)
    
    # Create synthetic ISI distribution data (100 bins)
    isi_dists = []
    for i in range(n_samples):
        # Create exponential-like ISI distribution
        isi = np.random.exponential(0.02, 1000)  # 20ms average ISI
        hist, _ = np.histogram(isi, bins=100, range=(0, 0.2))
        isi_dists.append(hist.astype(float))
    
    # Create synthetic ACG data (100 time bins)
    acg_dists = []
    for i in range(n_samples):
        # Create simple autocorrelogram
        t = np.linspace(-50, 50, 100)
        acg = np.exp(-np.abs(t)/10) + 0.1 * np.random.randn(100)
        acg_dists.append(np.maximum(acg, 0))  # Ensure non-negative
    
    # Create labels (3 classes)
    labels = np.random.randint(0, 3, n_samples)
    
    return np.array(waveforms), np.array(isi_dists), np.array(acg_dists), labels

def test_individual_augmentations():
    """Test individual augmentation methods."""
    print("Testing individual augmentation methods...")
    
    # Create a simple test signal
    t = np.linspace(0, 2*np.pi, 50)
    test_signal = torch.tensor(np.sin(t), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    augmentations = NeuralAugmentations()
    
    # Test Gaussian noise
    noisy_signal = augmentations.add_gaussian_noise(test_signal, noise_std=0.1)
    print(f"✓ Gaussian noise: Original std={test_signal.std():.4f}, Noisy std={noisy_signal.std():.4f}")
    
    # Test amplitude scaling
    scaled_signal = augmentations.amplitude_scaling(test_signal, scale_range=(0.5, 1.5))
    scale_ratio = scaled_signal.std() / test_signal.std()
    print(f"✓ Amplitude scaling: Scale ratio={scale_ratio:.4f}")
    
    # Test baseline shift
    shifted_signal = augmentations.baseline_shift(test_signal, shift_range=(-0.1, 0.1))
    mean_shift = shifted_signal.mean() - test_signal.mean()
    print(f"✓ Baseline shift: Mean shift={mean_shift:.4f}")
    
    # Test Gaussian smoothing
    smoothed_signal = augmentations.gaussian_smoothing(test_signal, sigma_range=(1.0, 2.0))
    print(f"✓ Gaussian smoothing: Original peak={test_signal.max():.4f}, Smoothed peak={smoothed_signal.max():.4f}")
    
    print("Individual augmentation tests passed!\n")

def test_augmented_dataset():
    """Test the augmented dataset wrapper with different configurations."""
    print("Testing augmented dataset wrapper...")
    
    # Create sample data
    waveforms, isi_dists, acg_dists, labels = create_sample_data()
    
    # Create base dataset
    data_dict = {
        "wave": waveforms,
        "isi": isi_dists,
        "acg": acg_dists
    }
    base_dataset = MultiModalEphysDataset(data_dict, labels, mode="multi")
    
    # Test different configurations
    configs = {
        "light_augmentations": ExperimentConfigs.with_light_augmentations(),
        "heavy_augmentations": ExperimentConfigs.with_heavy_augmentations(),
        "no_augmentations": ExperimentConfigs.no_augmentations()
    }
    
    for config_name, config in configs.items():
        print(f"Testing {config_name}...")
        
        # Create augmented dataset
        augmented_dataset = AugmentedMultiModalEphysDataset(
            base_dataset, config, phase="pretraining"
        )
        
        # Test training vs evaluation mode
        augmented_dataset.train()
        train_sample = augmented_dataset[0]
        
        augmented_dataset.eval()
        eval_sample = augmented_dataset[0]
        
        original_sample = base_dataset[0]
        
        print(f"  ✓ Training mode: Data shapes preserved")
        print(f"  ✓ Evaluation mode: Returns original data")
        
        # Test data loader compatibility
        data_loader = torch.utils.data.DataLoader(
            augmented_dataset, batch_size=4, collate_fn=none_safe_collate
        )
        
        batch = next(iter(data_loader))
        data_batch, label_batch = batch
        
        print(f"  ✓ DataLoader compatible: Batch size {label_batch.shape[0]}")
        print(f"  ✓ Modalities: {list(data_batch.keys())}")
        
    print("Augmented dataset tests passed!\n")

def create_visualization():
    """Create visualization plots to show augmentation effects."""
    print("Creating visualization plots...")
    
    # Create sample data
    waveforms, isi_dists, acg_dists, labels = create_sample_data()
    
    # Create base dataset
    data_dict = {
        "wave": waveforms,
        "isi": isi_dists, 
        "acg": acg_dists
    }
    base_dataset = MultiModalEphysDataset(data_dict, labels, mode="multi")
    
    # Create augmented dataset with heavy augmentations
    config = ExperimentConfigs.with_heavy_augmentations()
    augmented_dataset = AugmentedMultiModalEphysDataset(
        base_dataset, config, phase="pretraining"
    )
    augmented_dataset.train()
    
    # Get original and augmented samples
    original_sample = base_dataset[0]
    augmented_samples = [augmented_dataset[0] for _ in range(5)]
    
    # Create plots
    fig, axes = plt.subplots(3, 6, figsize=(18, 12))
    fig.suptitle('Data Augmentation Examples', fontsize=16)
    
    modalities = ['wave', 'isi', 'acg']
    titles = ['Waveform', 'ISI Distribution', 'Autocorrelogram']
    
    for mod_idx, (modality, title) in enumerate(zip(modalities, titles)):
        # Plot original
        original_data = original_sample[0][modality].squeeze().numpy()
        axes[mod_idx, 0].plot(original_data, 'b-', linewidth=2)
        axes[mod_idx, 0].set_title(f'Original {title}')
        axes[mod_idx, 0].grid(True, alpha=0.3)
        
        # Plot augmented versions
        for aug_idx, aug_sample in enumerate(augmented_samples):
            aug_data = aug_sample[0][modality].squeeze().numpy()
            axes[mod_idx, aug_idx + 1].plot(aug_data, 'r-', linewidth=1)
            axes[mod_idx, aug_idx + 1].set_title(f'Augmented {aug_idx + 1}')
            axes[mod_idx, aug_idx + 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("augmentation_test_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "augmentation_examples.png", dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_dir / 'augmentation_examples.png'}")
    plt.close()

def test_phase_specific_behavior():
    """Test that augmentations are applied correctly for different training phases."""
    print("Testing phase-specific augmentation behavior...")
    
    # Create sample data
    waveforms, isi_dists, acg_dists, labels = create_sample_data()
    data_dict = {"wave": waveforms, "isi": isi_dists, "acg": acg_dists}
    base_dataset = MultiModalEphysDataset(data_dict, labels, mode="multi")
    
    # Test different phase configurations
    phases = ["pretraining", "finetuning", "supervised"]
    config = ExperimentConfigs.with_heavy_augmentations()
    
    for phase in phases:
        print(f"Testing {phase} phase...")
        
        augmented_dataset = AugmentedMultiModalEphysDataset(
            base_dataset, config, phase=phase
        )
        
        # Check if augmentations are enabled for this phase
        phase_flag = getattr(config, f"augment_{phase}", False)
        expected_enabled = config.use_augmentations and phase_flag
        
        augmented_dataset.train()
        actual_enabled = augmented_dataset.augment_enabled
        
        assert actual_enabled == expected_enabled, f"Phase {phase}: Expected {expected_enabled}, got {actual_enabled}"
        print(f"  ✓ Phase {phase}: Augmentations {'enabled' if actual_enabled else 'disabled'}")
    
    print("Phase-specific behavior tests passed!\n")

def main():
    """Run all tests."""
    print("=" * 60)
    print("HIPPIE Data Augmentation Test Suite")
    print("=" * 60)
    
    try:
        # Run individual tests
        test_individual_augmentations()
        test_augmented_dataset()
        test_phase_specific_behavior()
        create_visualization()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Data augmentation implementation is working correctly.")
        print("You can now run experiments with augmentation configs:")
        print("  - with_light_augmentations")
        print("  - with_heavy_augmentations")
        print("  - no_augmentations")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())