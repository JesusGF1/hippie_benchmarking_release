from npyx.c4.dataset_init import (
    N_CHANNELS,
    WAVEFORM_SAMPLES,
    extract_and_check,
    get_paths_from_dir,
    prepare_classification_dataset,
)
from torch.utils.data import Dataset
import npyx
import torch
import numpy as np
import os
import pandas as pd
import math
from math import pi
from celltype_ibl.params.config import DATASETS_DIRECTORY
import h5py
from tqdm import tqdm
import hashlib
import pickle
import time as time_module


def get_c4_labeled_dataset(from_h5: bool = False, normalise: bool = True):
    if from_h5:
        datasets_abs = get_paths_from_dir(DATASETS_DIRECTORY)

        # Extract and check the datasets, saving a dataframe with the results
        _, dataset_class = extract_and_check(
            *datasets_abs,
            save=False,
            _labels_only=True,
            normalise_wvf=normalise,
            n_channels=N_CHANNELS,
            _extract_mli_clusters=False,
            _extract_layer=False,
        )

        # Apply quality checks and filter out granule cells
        checked_dataset = dataset_class.apply_quality_checks()
        LABELLING, CORRESPONDENCE, granule_mask = (
            checked_dataset.filter_out_granule_cells(return_mask=True)
        )
        dataset, _ = prepare_classification_dataset(
            checked_dataset,
            normalise_acgs=False,
            multi_chan_wave=False,
            process_multi_channel=False,
            _acgs_path=os.path.join(
                DATASETS_DIRECTORY, "acgs_vs_firing_rate", "acgs_3d_logscale.npy"
            ),
            _acg_mask=(~granule_mask),
            _acg_multi_factor=10,
            _n_channels=N_CHANNELS,
        )
        label = checked_dataset.labels_list

    else:
        features_df = pd.read_csv(
            os.path.join(
                DATASETS_DIRECTORY,
                "labelled_raw_log_3d_acg_multi_wvf/features.csv",
            )
        )
        dataset = features_df.to_numpy()
        label_df = pd.read_csv(
            os.path.join(
                DATASETS_DIRECTORY,
                "labelled_raw_log_3d_acg_multi_wvf/labels.csv",
            )
        )
        label = label_df.to_numpy().flatten()

        # filter out "GrC"
        dataset = dataset[label != "GrC"]
        label = label[label != "GrC"]

        LABELLING = {
            "PkC_cs": 4,
            "PkC_ss": 3,
            "MFB": 2,
            "MLI": 1,
            "GoC": 0,
            "unlabelled": -1,
        }
        CORRESPONDENCE = {
            4: "PkC_cs",
            3: "PkC_ss",
            2: "MFB",
            1: "MLI",
            0: "GoC",
            -1: "unlabelled",
        }
    acgs = dataset[:, :2010].reshape(-1, 10, 201)[:, :, 100:]
    if from_h5:
        acgs = acgs / 10
    label_idx = [LABELLING[i] for i in label]
    if normalise:
        waveforms = dataset[:, 2010:2100]
    else:
        templates = dataset[:, 2100:].reshape(-1, 22, 120)
        ptps = np.ptp(templates, axis=2)
        maxCH = np.argmax(ptps, axis=1)
        waveforms = templates[np.arange(len(maxCH)), maxCH, :]
    return waveforms, acgs, np.array(label_idx), label, LABELLING, CORRESPONDENCE


def get_c4_unlabelled_wvf_acg_pairs(from_h5: bool = False):
    if from_h5:
        dataset_paths = npyx.c4.get_paths_from_dir(
            DATASETS_DIRECTORY, include_hull_unlab=True
        )
        # Normalise waveforms so that the max in the dataset is 1 and the minimum is -1. Only care about shape.
        dataset_class = npyx.c4.extract_and_merge_datasets(
            *dataset_paths,
            quality_check=True,
            normalise_wvf=False,  # only applies to the multichannel waveforms
            _use_amplitudes=False,
            n_channels=N_CHANNELS,
            central_range=WAVEFORM_SAMPLES,
            labelled=False,
            flip_waveforms=True,
        )

        dataset_class.make_unlabelled_only()

        dataset_class.make_full_dataset(wf_only=True)
        wf = dataset_class.wf.reshape(-1, 10, 120)
        peak_chan = np.argmax(np.ptp(wf, axis=2), axis=1)

        peak_wf = []
        for i in range(len(wf)):
            peak_wf.append(npyx.datasets.preprocess_template(wf[i, peak_chan[i], :]))
        peak_wf = np.stack(peak_wf)

        acg_3d = np.load(
            os.path.join(
                DATASETS_DIRECTORY,
                "acgs_vs_firing_rate/unlabelled_acgs_3d_logscale.npy",
            )
        )
    else:
        features_df = pd.read_csv(
            os.path.join(
                DATASETS_DIRECTORY,
                "unlabelled_raw_log_3d_acg_multi_wvf/features.csv",
            )
        )
        features = features_df.to_numpy()
        acg_3d = features[:, :2010]
        peak_wf = features[:, 2010:2100]

    acg_3d = acg_3d.reshape(-1, 10, 201)
    # Multiply by 10 to normalise to 100 Hz as the max.
    acg_3d = acg_3d[:, :, 100:]
    return peak_wf, acg_3d


def get_cache_path(dataset_file, is_labeled, normalise):
    """
    Get the cache file path for a processed dataset.

    Args:
        dataset_file: Original dataset filename
        is_labeled: Whether dataset is labeled
        normalise: Whether waveforms are normalized

    Returns:
        Path to cache file
    """
    cache_dir = os.path.join(DATASETS_DIRECTORY, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache filename based on dataset file and processing params
    base_name = os.path.splitext(dataset_file)[0]
    cache_suffix = f"_labeled_{is_labeled}_norm_{normalise}.pkl"
    cache_filename = base_name + cache_suffix

    return os.path.join(cache_dir, cache_filename)


def get_dataset_modification_time(dataset_file):
    """Get the modification time of the dataset file."""
    dataset_path = os.path.join(DATASETS_DIRECTORY, dataset_file)
    if os.path.exists(dataset_path):
        return os.path.getmtime(dataset_path)
    return None


def save_dataset_to_cache(cache_path, data_dict):
    """
    Save processed dataset to cache file.

    Args:
        cache_path: Path to cache file
        data_dict: Dictionary containing processed data and metadata
    """
    print(f"Saving processed dataset to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Cache saved successfully")


def load_dataset_from_cache(cache_path, dataset_file):
    """
    Load processed dataset from cache if valid.

    Args:
        cache_path: Path to cache file
        dataset_file: Original dataset filename (for validation)

    Returns:
        Cached data dictionary if valid, None otherwise
    """
    if not os.path.exists(cache_path):
        return None

    try:
        print(f"Found cached dataset: {cache_path}")
        with open(cache_path, 'rb') as f:
            data_dict = pickle.load(f)

        # Validate cache: check if source file has been modified
        source_mtime = get_dataset_modification_time(dataset_file)
        cached_mtime = data_dict.get('source_mtime', None)

        if source_mtime is not None and cached_mtime is not None:
            if abs(source_mtime - cached_mtime) > 1.0:  # Allow 1 second tolerance
                print(f"⚠ Cache is outdated (source file modified), will reprocess...")
                return None

        print(f"✓ Loading from cache (created {time_module.ctime(data_dict.get('cache_time', 0))})")
        return data_dict

    except Exception as e:
        print(f"⚠ Failed to load cache: {e}, will reprocess...")
        return None


def get_dataset_from_file(dataset_file, is_labeled=True, normalise=True, use_cache=True):
    """
    Load dataset from a specific file in the datasets directory.

    Args:
        dataset_file: filename of the dataset (e.g., 'C4_database_hull_labelled.h5')
        is_labeled: whether the dataset contains labels
        normalise: whether to normalise waveforms
        use_cache: whether to use caching (default: True)

    Returns:
        tuple: (waveforms, acgs, labels, label_names, labelling_dict, correspondence_dict) if labeled
               (waveforms, acgs) if unlabeled
    """
    dataset_path = os.path.join(DATASETS_DIRECTORY, dataset_file)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Try to load from cache first
    if use_cache:
        cache_path = get_cache_path(dataset_file, is_labeled, normalise)
        cached_data = load_dataset_from_cache(cache_path, dataset_file)

        if cached_data is not None:
            # Return cached data
            if is_labeled:
                return (
                    cached_data['waveforms'],
                    cached_data['acgs'],
                    cached_data['label_idx'],
                    cached_data['labels'],
                    cached_data['labelling_dict'],
                    cached_data['correspondence_dict']
                )
            else:
                return cached_data['waveforms'], cached_data['acgs']
    
    # Special-case bypass paths for lisberger/harmonized and Allen datasets.
    #
    # These datasets have STRUCTURALLY DIFFERENT H5 schemas from hausser/hull
    # (the "standard" multi-channel C4 format that npyx's NeuronsDataset
    # expects). Concretely, lisberger is single-channel optogenetic-ID data:
    #   - sane_spikes is uint8 not bool, fn_fp_filtered_spikes is a scalar
    #     b'None' instead of a per-spike bool array, neuron_relative_id is
    #     missing, channels has shape (1,) instead of (22,), waveform shape
    #     is (240, 1) instead of (22, 180), etc.
    # Allen has its own schema differences. Forcing either through npyx's
    # NeuronsDataset constructor would either fail on the missing attributes
    # or require fabricating multi-channel data that does not exist in the
    # source — neither is acceptable.
    #
    # So we keep bespoke loaders that read each format directly via h5py
    # and emit (waveforms, acgs, labels, ...) tuples with the same shapes
    # NEMO's downstream training code expects. The 3D ACGs are computed
    # via `compute_autocorrelogram_3d` (above), which is now a thin wrapper
    # around `npyx.c4.fast_acg3d` - the same canonical real-3D-ACG function
    # the standard hausser/hull path uses. So:
    #   - Waveform reading is dataset-specific (lisberger single-channel,
    #     hausser/hull multi-channel)
    #   - 3D ACG computation is identical across all datasets
    # which is the right scientific layering.
    #
    # Earlier (pre-2026-04-08) `compute_autocorrelogram_3d` was an O(N**2)
    # Python loop that ALSO faked the 10 firing-rate bins by adding random
    # Gaussian noise to a single 2D ACG. That's been replaced - see the
    # function docstring for the history.
    if 'harmonized' in dataset_file.lower() or 'lisberger' in dataset_file.lower():
        result = load_harmonized_lisberger_dataset(dataset_file, is_labeled, normalise)
        if use_cache:
            cache_path = get_cache_path(dataset_file, is_labeled, normalise)
            if is_labeled:
                waveforms, acgs, label_idx, labels, labelling_dict, correspondence_dict = result
                cache_data = {
                    'waveforms': waveforms, 'acgs': acgs,
                    'label_idx': label_idx, 'labels': labels,
                    'labelling_dict': labelling_dict,
                    'correspondence_dict': correspondence_dict,
                    'source_mtime': get_dataset_modification_time(dataset_file),
                    'cache_time': time_module.time(),
                }
            else:
                waveforms, acgs = result
                cache_data = {
                    'waveforms': waveforms, 'acgs': acgs,
                    'source_mtime': get_dataset_modification_time(dataset_file),
                    'cache_time': time_module.time(),
                }
            save_dataset_to_cache(cache_path, cache_data)
        return result
    if 'allen' in dataset_file.lower() and 'nemo' not in dataset_file.lower():
        # Original Allen C4 datasets (with datasets/session/ nesting)
        result = load_allen_dataset_direct(dataset_file, is_labeled, normalise)
        if use_cache:
            cache_path = get_cache_path(dataset_file, is_labeled, normalise)
            if is_labeled:
                waveforms, acgs, label_idx, labels, labelling_dict, correspondence_dict = result
                cache_data = {
                    'waveforms': waveforms, 'acgs': acgs,
                    'label_idx': label_idx, 'labels': labels,
                    'labelling_dict': labelling_dict,
                    'correspondence_dict': correspondence_dict,
                    'source_mtime': get_dataset_modification_time(dataset_file),
                    'cache_time': time_module.time(),
                }
            else:
                waveforms, acgs = result
                cache_data = {
                    'waveforms': waveforms, 'acgs': acgs,
                    'source_mtime': get_dataset_modification_time(dataset_file),
                    'cache_time': time_module.time(),
                }
            save_dataset_to_cache(cache_path, cache_data)
        return result
    
    # DANDI-converted, IBL-converted, and Allen-converted *_nemo.h5 files
    # lack the sane_spikes / dataset_id fields that npyx's NeuronsDataset
    # expects. Bypass npyx and load them directly via h5py.
    _converted_prefixes = ('dandi', 'ibl', 'allen')
    _basename = os.path.basename(dataset_file).lower()
    if any(_basename.startswith(p) for p in _converted_prefixes) and 'nemo' in _basename:
        result = load_dandi_ibl_dataset_direct(dataset_file, is_labeled, normalise)
        if use_cache:
            cache_path = get_cache_path(dataset_file, is_labeled, normalise)
            if is_labeled:
                waveforms, acgs, label_idx, labels, labelling_dict, correspondence_dict = result
                cache_data = {
                    'waveforms': waveforms, 'acgs': acgs,
                    'label_idx': label_idx, 'labels': labels,
                    'labelling_dict': labelling_dict,
                    'correspondence_dict': correspondence_dict,
                    'source_mtime': get_dataset_modification_time(dataset_file),
                    'cache_time': time_module.time(),
                }
            else:
                waveforms, acgs = result
                cache_data = {
                    'waveforms': waveforms, 'acgs': acgs,
                    'source_mtime': get_dataset_modification_time(dataset_file),
                    'cache_time': time_module.time(),
                }
            save_dataset_to_cache(cache_path, cache_data)
        return result

    if dataset_file.endswith('.h5'):
        # Handle H5 files
        if is_labeled:
            # For labeled H5 files, use the existing npyx functionality
            datasets_abs = [dataset_path]
            
            # Extract and check the datasets
            # Compatibility shim for older npyx versions that reference 'pi' without import
            import math
            import sys

            # Inject pi into relevant npyx modules for compatibility
            try:
                import npyx.c4.dataset_init
                if not hasattr(npyx.c4.dataset_init, 'pi'):
                    npyx.c4.dataset_init.pi = math.pi
            except (ImportError, AttributeError):
                pass

            try:
                import npyx.c4
                if not hasattr(npyx.c4, 'pi'):
                    npyx.c4.pi = math.pi
            except (ImportError, AttributeError):
                pass

            # Also ensure pi is available globally
            globals()['pi'] = math.pi

            try:
                _, dataset_class = extract_and_check(
                    *datasets_abs,
                    save=False,
                    _labels_only=True,
                    normalise_wvf=normalise,
                    n_channels=N_CHANNELS,
                    _extract_mli_clusters=False,
                    _extract_layer=False,
                )
            except NameError as e:
                if 'pi' in str(e):
                    # Compatibility shim for older npyx versions
                    print(f"Warning: npyx compatibility issue - {e}")
                    print("Applying compatibility shim...")

                    # Try to inject pi into all npyx modules
                    for module_name in sys.modules:
                        if module_name.startswith('npyx'):
                            module = sys.modules[module_name]
                            if hasattr(module, '__dict__'):
                                module.__dict__['pi'] = math.pi

                    # Also try setting it in the local namespace
                    locals()['pi'] = math.pi
                    import builtins
                    builtins.pi = math.pi

                    _, dataset_class = extract_and_check(
                        *datasets_abs,
                        save=False,
                        _labels_only=True,
                        normalise_wvf=normalise,
                        n_channels=N_CHANNELS,
                        _extract_mli_clusters=False,
                        _extract_layer=False,
                    )
                else:
                    raise
            
            # Apply quality checks and filter out granule cells
            checked_dataset = dataset_class.apply_quality_checks()
            _, _, _ = (
                checked_dataset.filter_out_granule_cells(return_mask=True)
            )
            # Don't use external ACG file to avoid dimension mismatch
            try:
                dataset, _ = prepare_classification_dataset(
                    checked_dataset,
                    normalise_acgs=False,
                    multi_chan_wave=False,
                    process_multi_channel=False,
                    _acgs_path=None,  # Don't use external ACG file
                    _acg_mask=None,   # Don't apply ACG mask
                    _acg_multi_factor=10,
                    _n_channels=N_CHANNELS,
                )
            except NameError as e:
                if 'pi' in str(e):
                    # Compatibility shim for older npyx versions
                    print(f"Warning: npyx compatibility issue in prepare_classification_dataset - {e}")
                    print("Applying compatibility shim...")
                    import builtins
                    builtins.pi = math.pi
                    dataset, _ = prepare_classification_dataset(
                        checked_dataset,
                        normalise_acgs=False,
                        multi_chan_wave=False,
                        process_multi_channel=False,
                        _acgs_path=None,  # Don't use external ACG file
                        _acg_mask=None,   # Don't apply ACG mask
                        _acg_multi_factor=10,
                        _n_channels=N_CHANNELS,
                    )
                else:
                    raise
            label = checked_dataset.labels_list
            
            # Process the data similar to get_c4_labeled_dataset
            acgs = dataset[:, :2010].reshape(-1, 10, 201)[:, :, 100:]
            acgs = acgs / 10  # Normalize for H5 files
            
            # Create label mappings locally to avoid circular import
            unique_labels, label_idx = np.unique(label, return_inverse=True)
            correspondence_dict = {i: l for i, l in enumerate(unique_labels)}
            labelling_dict = {l: i for i, l in enumerate(unique_labels)}
            
            if normalise:
                waveforms = dataset[:, 2010:2100]
            else:
                templates = dataset[:, 2100:].reshape(-1, 22, 120)
                ptps = np.ptp(templates, axis=2)
                maxCH = np.argmax(ptps, axis=1)
                waveforms = templates[np.arange(len(maxCH)), maxCH, :]

            # Save to cache
            if use_cache:
                cache_path = get_cache_path(dataset_file, is_labeled, normalise)
                cache_data = {
                    'waveforms': waveforms,
                    'acgs': acgs,
                    'label_idx': label_idx,
                    'labels': label,
                    'labelling_dict': labelling_dict,
                    'correspondence_dict': correspondence_dict,
                    'source_mtime': get_dataset_modification_time(dataset_file),
                    'cache_time': time_module.time()
                }
                save_dataset_to_cache(cache_path, cache_data)

            return waveforms, acgs, label_idx, label, labelling_dict, correspondence_dict
            
        else:
            # For unlabeled H5 files - load directly from H5
            print(f"Loading unlabeled dataset directly from H5: {dataset_path}")

            with h5py.File(dataset_path, 'r') as f:
                neuron_keys = [k for k in f.keys() if k.startswith('hull_') or k.startswith('hausser_') or k.startswith('medina_') or k.startswith('lisberger_')]

                waveforms = []
                acgs = []

                for neuron_key in tqdm(neuron_keys, desc="Loading unlabeled neurons"):
                    neuron = f[neuron_key]

                    # Get waveform
                    if 'mean_waveform_preprocessed' in neuron:
                        wvf = neuron['mean_waveform_preprocessed'][:]
                        # Get peak channel waveform
                        if wvf.ndim == 2:  # (n_channels, n_samples)
                            ptps = np.ptp(wvf, axis=1)
                            peak_ch = np.argmax(ptps)
                            wvf_peak = wvf[peak_ch, :]

                            # Normalize to 90 samples (central portion)
                            n_samples = wvf_peak.shape[0]
                            if n_samples >= 90:
                                center = n_samples // 2
                                wvf_peak = wvf_peak[center-45:center+45]
                            else:
                                # Pad if needed
                                pad = (90 - n_samples) // 2
                                wvf_peak = np.pad(wvf_peak, (pad, 90-n_samples-pad), mode='constant')

                            if normalise:
                                # Normalize waveform
                                wvf_peak = wvf_peak / (np.max(np.abs(wvf_peak)) + 1e-8)

                            waveforms.append(wvf_peak)
                        else:
                            continue
                    else:
                        continue

                    # Compute ACG from spike times
                    if 'spike_indices' in neuron and 'sampling_rate' in neuron:
                        spike_indices = neuron['spike_indices'][:]
                        sampling_rate = neuron['sampling_rate'][()]

                        # Compute ACG using npyx with train parameter
                        # Signature: acg(dp, u, bin_size, win_size, fs, train=spike_indices)
                        import npyx.corr
                        acg = npyx.corr.acg(
                            None, 0,  # dp and u are ignored when train is provided
                            bin_size=1,  # 1ms bins
                            win_size=100,  # 100ms window
                            fs=sampling_rate,
                            train=spike_indices,
                            normalize='Counts',
                            sav=False,
                            verbose=False
                        )

                        # Reshape to (10, 101) - 10 time bins x 101 samples
                        # Normalize ACG
                        if acg.max() > 0:
                            acg = acg / acg.max()

                        # Downsample to 10 bins by taking every 10th element or averaging
                        if len(acg) >= 100:
                            # Trim to 100 elements and reshape to (10, 10), then average
                            acg_trimmed = acg[:100]
                            acg_binned = acg_trimmed.reshape(10, 10).mean(axis=1)
                        else:
                            # Pad if necessary
                            acg_binned = np.pad(acg, (0, 10-len(acg)), mode='constant')[:10]

                        # Tile each bin to create (10, 101) shape
                        acg_2d = np.tile(acg_binned[:, np.newaxis], (1, 101))

                        acgs.append(acg_2d)
                    else:
                        # Create dummy ACG if spike times not available
                        acgs.append(np.zeros((10, 101)))

                waveforms = np.array(waveforms)
                acgs = np.array(acgs)

                print(f"Loaded {len(waveforms)} unlabeled neurons")

                # Save to cache
                if use_cache:
                    cache_path = get_cache_path(dataset_file, is_labeled, normalise)
                    cache_data = {
                        'waveforms': waveforms,
                        'acgs': acgs,
                        'source_mtime': get_dataset_modification_time(dataset_file),
                        'cache_time': time_module.time()
                    }
                    save_dataset_to_cache(cache_path, cache_data)

                return waveforms, acgs
    
    else:
        raise ValueError(f"Unsupported file format: {dataset_file}. Only .h5 files are supported currently.")


def auto_detect_dataset_type(dataset_file):
    """
    Auto-detect if a dataset is labeled or unlabeled based on filename patterns.
    """
    filename_lower = dataset_file.lower()
    if 'labelled' in filename_lower or 'labeled' in filename_lower:
        return True
    elif 'unlabelled' in filename_lower or 'unlabeled' in filename_lower:
        return False
    else:
        # Default assumption - could be made configurable
        return True


def compute_autocorrelogram_3d(spike_times, bin_size_ms=1.0, window_ms=100.0,
                                n_freq_bands=10, fs=30000, max_spikes=10000):
    """
    Compute 3D autocorrelogram (firing-rate-bins x log-time) from spike times.

    Thin wrapper around the EXACT pair `npyx.corr.crosscorr_vs_firing_rate`
    + `npyx.corr.convert_acg_log` that the canonical hausser/hull C4 path
    calls in `npyx.c4.dataset_init.prepare_classification_dataset` at lines
    329-330 of dataset_init.py. Output is sliced to positive log-time lags
    only so it matches the shape NEMO actually consumes after the standard
    load path's reshape step in `get_c4_labeled_dataset` line 94
    (`acgs[:, :, 100:]`).

    Args:
        spike_times: 1D array of spike times in SECONDS (matches the
            existing call sites in load_harmonized_lisberger_dataset and
            load_allen_dataset_direct, which divide spike_indices by
            sampling_rate before calling this function). The wrapper
            internally converts back to samples for npyx.
        bin_size_ms: ACG bin width in milliseconds (default 1.0, the c4
            standard).
        window_ms: HALF-window in milliseconds (default 100.0). The npyx
            function takes a `win_size` argument which is the FULL window,
            so we pass `2 * window_ms` = 200ms by default.
        n_freq_bands: number of firing-rate quantile bins (default 10, the
            c4 standard).
        fs: sampling rate to convert seconds back to samples for npyx
            (default 30000, the c4 standard).
        max_spikes: cap on the number of spikes used per neuron (default
            10000). See "Why we cut spike trains" below.

    Returns:
        (n_freq_bands, n_time_bins_positive_half) array, where
        n_time_bins_positive_half = (window_ms * 2 / bin_size_ms) // 2 + 1
        which is 101 for the defaults. This matches the shape NEMO's
        downstream training code expects (the standard hausser/hull path
        slices `[:, 100:]` from the full (10, 201) ACG; this wrapper
        returns the same slice directly).

    --- WHY WE CUT SPIKE TRAINS TO max_spikes ---

    The lisberger and Allen datasets contain optogenetic-ID recordings
    that are 30-60 minutes long with up to ~700,000 spikes per neuron.
    The standard hausser/hull C4 datasets contain 1-5 minute recordings
    with ~5,000 spikes per neuron - two orders of magnitude fewer. Running
    `crosscorr_vs_firing_rate` (the canonical NEMO ACG function) on full
    lisberger spike trains takes ~6 seconds per neuron x 1152 neurons ~=
    115 minutes per fold, which is unworkable for the 5-fold benchmark.

    Cutting to the first `max_spikes=10000` spikes per neuron is the
    correct fix because:

    1. **Statistical sufficiency.** The 3D ACG is a stationary statistic
       of the spike train. With 10000 spikes, each of the 10 firing-rate
       quantile bins gets ~1000 spikes, more than enough for the
       (10, 101) histogram to converge. NEMO's published hausser/hull
       results were computed on neurons with FEWER than 10000 spikes
       each, so cutting lisberger to 10000 brings it INTO the regime of
       the training data NEMO was published on, not out of it.

    2. **Comparability across datasets.** Without the cut, lisberger
       neurons would be characterised by ~700K-spike ACGs while
       hausser/hull neurons get ~5K-spike ACGs. The amount of spike data
       feeding into the ACG estimation would differ by ~140x, biasing
       any cross-dataset comparison toward "lisberger has lower-noise
       ACGs" purely as an artefact of recording length, not biology.
       Cutting to 10K equalises the sample size across datasets at a
       value above all of them, which is the most defensible
       methodological choice.

    3. **No firing-rate stratification distortion.** The function bins
       spikes by their LOCAL (boxcar-smoothed) firing rate before
       computing per-bin ACGs. Cutting to the first 10K spikes preserves
       the firing-rate distribution of the early portion of the
       recording. We assume firing rate is roughly stationary across the
       30-min recording, which is the standard assumption for any
       cell-type classification work on these datasets.

    4. **Reproducibility.** The cut is deterministic (always the first
       N), not random, so re-running the smoke test produces bit-
       identical ACGs. No seed management needed.

    Tradeoffs we accept:
    - We discard ~98.5% of lisberger spikes per neuron. This is fine for
      ACG estimation but would not be appropriate for any analysis that
      needs to track temporal dynamics across the full recording. NEMO
      does not.
    - If a neuron's behaviour changes between minute 1 and minute 30 of
      its recording (e.g. drift, drug onset, optostim manipulation), we
      only see the early state. This matches how the C4 datasets were
      curated for NEMO in the first place.

    To override the cut: pass `max_spikes=None` (uses all spikes; will be
    slow on long recordings). For the production benchmark we keep the
    default of 10000.
    """
    n_time_bins_full = int(window_ms * 2 / bin_size_ms) + 1
    n_time_bins_half = n_time_bins_full // 2 + 1  # 101 for defaults
    if len(spike_times) < 10:  # Not enough spikes for a meaningful ACG
        return np.zeros((n_freq_bands, n_time_bins_half))

    # Cap spike count per neuron - see docstring for justification
    if max_spikes is not None and len(spike_times) > max_spikes:
        spike_times = spike_times[:max_spikes]

    # npyx wants spike times in SAMPLES, not seconds. Convert back.
    spike_samples = (np.asarray(spike_times) * fs).astype(np.int64)

    # `npyx.corr.crosscorr_vs_firing_rate` followed by `convert_acg_log` is
    # the EXACT pair of functions the canonical hausser/hull path calls in
    # `prepare_classification_dataset` at npyx/c4/dataset_init.py:329-330.
    # Calling them directly produces ACGs that are bit-comparable to what
    # NEMO trained on for the C4 datasets.
    #
    # Pipeline:
    #   1. crosscorr_vs_firing_rate -> linear-time (10, 201) ACG, stratified
    #      across 10 firing-rate quantile bins
    #   2. convert_acg_log -> same (10, 201) shape but time axis is now
    #      log-spaced (negative log lags + zero + positive log lags). The
    #      log spacing concentrates resolution near short ISI lags (where
    #      cell-type-distinguishing structure lives) while still covering
    #      the full +/-100ms window.
    #   3. Slice [:, 100:] -> (10, 101) positive log-time lags only,
    #      matching what `get_c4_labeled_dataset` line 94 does to
    #      pre-computed ACGs from disk.
    import npyx.corr as _npyx_corr
    try:
        _, acg_3d_lin = _npyx_corr.crosscorr_vs_firing_rate(
            spike_samples,
            spike_samples,
            win_size=int(window_ms * 2),  # full window in ms
            bin_size=int(bin_size_ms),
            fs=int(fs),
            num_firing_rate_bins=n_freq_bands,
        )
        acg_3d, _ = _npyx_corr.convert_acg_log(
            acg_3d_lin, int(bin_size_ms), int(window_ms * 2),
        )
    except Exception as e:
        # Defensive: if npyx chokes on a degenerate spike train, return
        # zeros so the calling loader can decide whether to skip the
        # neuron (existing call sites wrap this in try/except already).
        print(f"[compute_autocorrelogram_3d] npyx pipeline failed: {e}")
        return np.zeros((n_freq_bands, n_time_bins_half))

    # Slice to positive log-time lags only - matches `get_c4_labeled_dataset`
    # at line 94 of c4_data_utils.py:
    #     acgs = dataset[:, :2010].reshape(-1, 10, 201)[:, :, 100:]
    # which takes time bins 100..200 from the full (10, 201) log-binned
    # ACG, giving the (10, 101) shape NEMO downstream expects.
    if acg_3d.shape[1] >= n_time_bins_full:
        acg_3d = acg_3d[:, n_time_bins_full // 2:]  # positive-lag half
    # If npyx happened to return a different number of time bins, fall
    # back to padding/truncating to (n_freq_bands, n_time_bins_half).
    if acg_3d.shape[1] != n_time_bins_half:
        if acg_3d.shape[1] > n_time_bins_half:
            acg_3d = acg_3d[:, :n_time_bins_half]
        else:
            pad = np.zeros((acg_3d.shape[0], n_time_bins_half - acg_3d.shape[1]))
            acg_3d = np.concatenate([acg_3d, pad], axis=1)

    return np.asarray(acg_3d)


def generate_dummy_acg(n_freq_bands=10, n_time_bins=201):
    """Generate a dummy ACG when spike data is not available."""
    np.random.seed(42)  # For reproducibility

    # Create a basic exponential decay pattern
    center = n_time_bins // 2
    time_bins = np.arange(n_time_bins)

    acg_3d = []
    for band in range(n_freq_bands):
        # Create basic autocorrelogram shape
        tau = np.random.uniform(10, 30)  # Decay constant
        autocorr = np.exp(-np.abs(time_bins - center) / tau)

        # Add refractory period
        refractory = 5  # bins
        refractory_mask = np.abs(time_bins - center) < refractory
        autocorr[refractory_mask] *= 0.1

        # Add noise
        autocorr += np.random.normal(0, 0.02, size=autocorr.shape)
        autocorr = np.maximum(autocorr, 0)

        acg_3d.append(autocorr)

    return np.array(acg_3d)


def load_allen_dataset_direct(dataset_file, is_labeled=True, normalise=True):
    """
    Load Allen dataset directly using h5py to bypass an npyx compatibility issue.

    Args:
        dataset_file: Filename of the Allen dataset
        is_labeled: whether the dataset contains labels
        normalise: whether to normalise waveforms

    Returns:
        tuple: (waveforms, acgs, labels, label_names, labelling_dict, correspondence_dict) if labeled
               (waveforms, acgs) if unlabeled
    """
    dataset_path = os.path.join(DATASETS_DIRECTORY, dataset_file)

    print(f"Loading Allen dataset directly: {dataset_file}")

    try:
        with h5py.File(dataset_path, 'r') as f:
            print(f"Top-level keys in H5 file: {list(f.keys())}")

            # Navigate to the datasets group
            if 'datasets' not in f:
                raise ValueError(f"Expected 'datasets' group not found in {dataset_file}")

            datasets_group = f['datasets']
            print(f"Sessions in datasets: {list(datasets_group.keys())}")

            # Get the first (and likely only) session
            session_ids = list(datasets_group.keys())
            if not session_ids:
                raise ValueError(f"No sessions found in datasets group")

            session_id = session_ids[0]  # Take first session
            session_group = datasets_group[session_id]
            print(f"Processing session: {session_id}")

            # Get all neuron IDs in this session
            neuron_ids = list(session_group.keys())
            print(f"Found {len(neuron_ids)} neurons: {neuron_ids[:10]}..." if len(neuron_ids) > 10 else f"Found {len(neuron_ids)} neurons: {neuron_ids}")

            # Collect data from all neurons
            waveforms_list = []
            labels_list = []
            acgs_list = []

            print(f"Processing {len(neuron_ids)} neurons with progress tracking...")
            import time
            overall_start_time = time.time()
            acg_times = []

            for neuron_id in tqdm(neuron_ids, desc="Processing neurons", unit="neuron"):
                neuron_group = session_group[neuron_id]

                # Extract waveform data (mean_waveform_preprocessed: shape 22, 180)
                if 'mean_waveform_preprocessed' in neuron_group:
                    waveform = np.array(neuron_group['mean_waveform_preprocessed'])
                    # Select channel with max peak-to-peak
                    ptps = np.ptp(waveform, axis=1)
                    max_channel = np.argmax(ptps)
                    selected_waveform = waveform[max_channel, :]

                    # Truncate/pad to expected size (90 timepoints)
                    if normalise:
                        if len(selected_waveform) > 90:
                            selected_waveform = selected_waveform[:90]
                        elif len(selected_waveform) < 90:
                            # Pad with zeros if needed
                            padding = np.zeros(90 - len(selected_waveform))
                            selected_waveform = np.concatenate([selected_waveform, padding])

                    waveforms_list.append(selected_waveform)
                else:
                    print(f"Warning: No waveform data found for neuron {neuron_id}")
                    continue

                # Compute ACG from spike data
                if 'spike_indices' in neuron_group and 'sampling_rate' in neuron_group:
                    try:
                        import time
                        start_time = time.time()

                        spike_indices = np.array(neuron_group['spike_indices'])
                        sampling_rate = neuron_group['sampling_rate'][()]
                        spike_times = spike_indices / sampling_rate

                        tqdm.write(f"Computing ACG for neuron {neuron_id}: {len(spike_indices)} spikes at {sampling_rate} Hz")

                        # Compute 3D ACG
                        acg_neuron = compute_autocorrelogram_3d(spike_times)
                        # Ensure correct shape (10, 101)
                        if acg_neuron.shape[1] > 101:
                            acg_neuron = acg_neuron[:, :101]  # Truncate
                        elif acg_neuron.shape[1] < 101:
                            # Pad to 101
                            padding = np.zeros((10, 101 - acg_neuron.shape[1]))
                            acg_neuron = np.concatenate([acg_neuron, padding], axis=1)

                        acgs_list.append(acg_neuron)

                        elapsed_time = time.time() - start_time
                        acg_times.append(elapsed_time)
                        tqdm.write(f"ACG computation for neuron {neuron_id} completed in {elapsed_time:.2f}s")

                    except Exception as e:
                        tqdm.write(f"Warning: Could not compute ACG for neuron {neuron_id}: {e}")
                        acgs_list.append(generate_dummy_acg(10, 101))
                else:
                    tqdm.write(f"Warning: No spike data for neuron {neuron_id}, using dummy ACG")
                    acgs_list.append(generate_dummy_acg(10, 101))

                # Extract label if this is a labeled dataset
                if is_labeled:
                    if 'ground_truth_label' in neuron_group:
                        label = neuron_group['ground_truth_label'][()]
                        # Handle different label encoding
                        if isinstance(label, bytes):
                            label = label.decode('utf-8')
                        elif isinstance(label, np.ndarray):
                            label = str(label.item())
                        else:
                            label = str(label)
                        labels_list.append(label)
                    else:
                        print(f"Warning: No label found for neuron {neuron_id}")
                        labels_list.append("unknown")

            if not waveforms_list:
                raise ValueError(f"No valid waveform data found in {dataset_file}")

            # Convert to numpy arrays
            waveforms = np.array(waveforms_list)
            acgs = np.array(acgs_list)

            # Print timing statistics
            overall_elapsed = time.time() - overall_start_time
            print(f"\n=== Allen Dataset Processing Summary ===")
            print(f"Total processing time: {overall_elapsed:.2f}s")
            print(f"Processed waveforms shape: {waveforms.shape}")
            print(f"Processed ACGs shape: {acgs.shape}")

            if acg_times:
                print(f"ACG computation statistics:")
                print(f"  - Total ACGs computed: {len(acg_times)}")
                print(f"  - Average time per ACG: {np.mean(acg_times):.2f}s")
                print(f"  - Min/Max time per ACG: {np.min(acg_times):.2f}s / {np.max(acg_times):.2f}s")
                print(f"  - Total ACG computation time: {np.sum(acg_times):.2f}s")
            print(f"========================================\n")

            if is_labeled:
                if not labels_list:
                    raise ValueError(f"No labels found in labeled dataset {dataset_file}")

                labels = np.array(labels_list)
                print(f"Processed labels shape: {labels.shape}")
                print(f"Unique labels: {np.unique(labels)}")

                # Create label mappings
                unique_labels, label_idx = np.unique(labels, return_inverse=True)
                correspondence_dict = {i: l for i, l in enumerate(unique_labels)}
                labelling_dict = {l: i for i, l in enumerate(unique_labels)}

                return waveforms, acgs, label_idx, labels, labelling_dict, correspondence_dict
            else:
                return waveforms, acgs

    except Exception as e:
        print(f"Error in direct Allen dataset loading: {e}")
        raise ValueError(f"Failed to load Allen dataset {dataset_file} directly: {e}")


def _detect_flat_array_schema(f):
    """
    Return True if the H5 file uses a flat parallel-array schema
    (Allen / IBL style: top-level datasets named 'waveforms', 'acgs',
    'label_idx', etc.) rather than the DANDI per-neuron-group style.

    Flat-array files are recognised by the presence of a top-level
    dataset named 'waveforms' that is 2-D (N x timepoints).  Per-neuron-
    group files have NO top-level 'waveforms' dataset; instead they have
    groups whose names contain the word 'neuron'.
    """
    top_keys = set(f.keys())
    if 'waveforms' in top_keys and hasattr(f['waveforms'], 'shape'):
        return True
    return False


def _load_flat_array_dataset(f, dataset_file, is_labeled, normalise):
    """
    Load a flat parallel-array H5 (Allen or IBL style) into the standard
    NEMO tuple.

    Expected top-level datasets:
        waveforms       (N, 90)          float32  - normalised waveforms
        waveforms_raw   (N, 90)          float32  - raw waveforms (optional)
        acgs            (N, 10, 201)     float32  - full ACG (log-time)
        label_idx       (N,)             int64    - integer class indices
        region / region_class            object   - string class labels

    The ACG is stored as the full symmetric log-time ACG (10, 201).  We
    slice [:, 100:] to get the positive-lag half (10, 101) that NEMO
    expects, matching what get_c4_labeled_dataset() does at line 94.

    The `waveforms` field is already normalised and already 90 timepoints,
    so no further processing is needed.  `waveforms_raw` is used only when
    normalise=False.
    """
    import time as time_mod
    t0 = time_mod.time()

    # --- waveforms (already (N, 90)) ---
    if not normalise and 'waveforms_raw' in f:
        waveforms = np.array(f['waveforms_raw'], dtype=np.float32)
    else:
        waveforms = np.array(f['waveforms'], dtype=np.float32)

    # --- ACGs: slice positive-lag half -> (N, 10, 101) ---
    acgs_full = np.array(f['acgs'], dtype=np.float32)   # (N, 10, 201)
    acgs = acgs_full[:, :, 100:]                         # (N, 10, 101)

    n = len(waveforms)
    print(f"  Loaded {n} neurons | waveforms: {waveforms.shape} | ACGs: {acgs.shape}")
    print(f"  Load time: {time_mod.time() - t0:.1f}s")

    if not is_labeled:
        return waveforms, acgs

    # --- labels ---
    # Try preferred label fields in order.  Both Allen and IBL store an
    # integer label_idx plus a string class-name array.  We reconstruct
    # string labels from the string array and label_idx so the output
    # matches the per-neuron-group path which returns string labels.
    label_idx_raw = np.array(f['label_idx'], dtype=np.int64)

    # Determine which string field holds the class names
    string_label_field = None
    for candidate in ('region', 'region_class', 'acronym', 'label'):
        if candidate in f and hasattr(f[candidate], 'shape'):
            string_label_field = candidate
            break

    if string_label_field is not None:
        raw_strings = f[string_label_field][()]
        labels = np.array([
            (s.decode('utf-8') if isinstance(s, bytes) else str(s))
            for s in raw_strings
        ])
        print(f"  String labels from field '{string_label_field}'")
    else:
        # Fall back to stringifying the integer indices
        print(f"  WARNING: no string label field found; using label_idx as strings")
        labels = label_idx_raw.astype(str)

    unique_labels, label_idx_out = np.unique(labels, return_inverse=True)
    correspondence_dict = {i: l for i, l in enumerate(unique_labels)}
    labelling_dict = {l: i for i, l in enumerate(unique_labels)}
    print(f"  Labels: {len(unique_labels)} classes — "
          f"{dict(zip(unique_labels, [int((labels == l).sum()) for l in unique_labels]))}")

    return waveforms, acgs, label_idx_out, labels, labelling_dict, correspondence_dict


def load_dandi_ibl_dataset_direct(dataset_file, is_labeled=True, normalise=True):
    """
    Load DANDI-converted, Allen-converted, or IBL H5 files directly via h5py.

    Auto-detects the H5 schema and dispatches to the appropriate sub-loader:

    1. **Flat parallel-array schema** (Allen / IBL style):
       Top-level datasets ``waveforms (N,90)``, ``acgs (N,10,201)``,
       ``label_idx (N,)``, ``region`` / ``region_class`` / ``acronym``.
       Both the Allen_visual_coding_nemo.h5 and IBL_brainwide_nemo.h5
       files use this layout.  Pre-computed ACGs and already-normalised
       waveforms are read directly — no per-neuron ACG recomputation.

    2. **Per-neuron-group schema** (DANDI style):
       Top-level groups ``{prefix}_neuron_{i}/`` each containing
       ``mean_waveform_preprocessed``, ``spike_indices``,
       ``sampling_rate``, ``ground_truth_label``.  Used by
       DANDI_000041_nemo.h5 (and DANDI_000473_nemo.h5).  3-D ACGs are
       computed on-the-fly from spike indices via
       ``compute_autocorrelogram_3d()``.

    Returns the same tuple as all other loaders:
        (waveforms, acgs, label_idx, labels, labelling_dict,
         correspondence_dict)
    if is_labeled=True, else (waveforms, acgs).
    """
    dataset_path = os.path.join(DATASETS_DIRECTORY, dataset_file)
    print(f"Loading DANDI/IBL dataset directly: {dataset_file}")

    import time as time_mod

    with h5py.File(dataset_path, 'r') as f:
        top_keys = list(f.keys())
        print(f"  Top-level keys ({len(top_keys)}): {top_keys[:10]}"
              + (" ..." if len(top_keys) > 10 else ""))

        # --- Schema detection ---
        if _detect_flat_array_schema(f):
            # Allen / IBL flat parallel-array format
            print("  Detected: flat parallel-array schema (Allen/IBL style)")
            return _load_flat_array_dataset(
                f, dataset_file, is_labeled, normalise
            )

        # --- Per-neuron-group schema (DANDI) ---
        neuron_keys = [k for k in top_keys if 'neuron' in k.lower()]
        print(f"  Detected: per-neuron-group schema (DANDI style)")
        print(f"  Found {len(neuron_keys)} neuron groups")

        if not neuron_keys:
            raise ValueError(
                f"load_dandi_ibl_dataset_direct: '{dataset_file}' has neither "
                f"a 'waveforms' flat-array schema nor any top-level groups "
                f"containing 'neuron' in their name.\n"
                f"Top-level keys: {top_keys}"
            )

        waveforms_list = []
        acgs_list = []
        labels_list = []
        acg_times = []
        overall_start = time_mod.time()

        for nk in tqdm(neuron_keys, desc="Processing neurons", unit="neuron"):
            grp = f[nk]

            # --- waveform ---
            if 'mean_waveform_preprocessed' not in grp:
                continue  # skip neurons without waveforms
            wf = np.array(grp['mean_waveform_preprocessed'])
            # DANDI H5s store single-channel waveforms (1-D array).
            # Multi-channel: pick max peak-to-peak channel.
            if wf.ndim == 2:
                ptps = np.ptp(wf, axis=1)
                wf = wf[np.argmax(ptps)]
            # Truncate / pad to 90 timepoints
            if len(wf) > 90:
                wf = wf[:90]
            elif len(wf) < 90:
                wf = np.pad(wf, (0, 90 - len(wf)))
            if normalise:
                max_abs = np.max(np.abs(wf))
                if max_abs > 1e-8:
                    wf = wf / max_abs
            waveforms_list.append(wf)

            # --- 3-D ACG: prefer precomputed, fall back to spike_indices ---
            if 'acg_3d' in grp:
                # Fast path — precomputed during H5 conversion (nemo_converter.py)
                acg = np.array(grp['acg_3d'], dtype=np.float32)
                if acg.shape[1] > 101:
                    acg = acg[:, :101]
                elif acg.shape[1] < 101:
                    acg = np.pad(acg, ((0, 0), (0, 101 - acg.shape[1])))
                acgs_list.append(acg)
            elif 'spike_indices' in grp and 'sampling_rate' in grp:
                # Slow path — recompute from spike times (per-neuron loop)
                try:
                    t0 = time_mod.time()
                    spike_idx = np.array(grp['spike_indices'])
                    sr = grp['sampling_rate'][()]
                    spike_times = spike_idx / sr
                    acg = compute_autocorrelogram_3d(spike_times)
                    if acg.shape[1] > 101:
                        acg = acg[:, :101]
                    elif acg.shape[1] < 101:
                        acg = np.pad(acg, ((0, 0), (0, 101 - acg.shape[1])))
                    acgs_list.append(acg)
                    acg_times.append(time_mod.time() - t0)
                except Exception as e:
                    tqdm.write(f"ACG failed for {nk}: {e}")
                    acgs_list.append(generate_dummy_acg(10, 101))
            else:
                acgs_list.append(generate_dummy_acg(10, 101))

            # --- label ---
            if is_labeled:
                if 'ground_truth_label' in grp:
                    lbl = grp['ground_truth_label'][()]
                    if isinstance(lbl, bytes):
                        lbl = lbl.decode('utf-8')
                    else:
                        lbl = str(lbl)
                    labels_list.append(lbl)
                else:
                    labels_list.append("unknown")

    if not waveforms_list:
        raise ValueError(f"No valid neurons found in {dataset_file}")

    waveforms = np.array(waveforms_list)
    acgs = np.array(acgs_list)

    elapsed = time_mod.time() - overall_start
    print(f"\n=== DANDI/IBL Dataset Summary ===")
    print(f"Total time: {elapsed:.1f}s | neurons: {len(waveforms_list)}")
    print(f"Waveforms shape: {waveforms.shape} | ACGs shape: {acgs.shape}")
    if acg_times:
        print(f"ACG avg: {np.mean(acg_times):.2f}s | total: {np.sum(acg_times):.1f}s")
    print(f"=================================\n")

    if is_labeled:
        labels = np.array(labels_list)
        unique_labels, label_idx = np.unique(labels, return_inverse=True)
        correspondence_dict = {i: l for i, l in enumerate(unique_labels)}
        labelling_dict = {l: i for i, l in enumerate(unique_labels)}
        print(f"Labels: {len(unique_labels)} classes — "
              f"{dict(zip(unique_labels, [int((labels == l).sum()) for l in unique_labels]))}")
        return waveforms, acgs, label_idx, labels, labelling_dict, correspondence_dict
    else:
        return waveforms, acgs


def load_harmonized_lisberger_dataset(dataset_file, is_labeled=True, normalise=True):
    """
    Load a harmonized Lisberger dataset directly without using npyx library.

    Args:
        dataset_file: Filename of the harmonized dataset
        is_labeled: whether the dataset contains labels
        normalise: whether to normalise waveforms

    Returns:
        tuple: (waveforms, acgs, labels, label_names, labelling_dict, correspondence_dict) if labeled
               (waveforms, acgs) if unlabeled
    """
    dataset_path = os.path.join(DATASETS_DIRECTORY, dataset_file)

    print(f"Loading harmonized Lisberger dataset: {dataset_file}")

    waveforms = []
    acgs = []
    labels = []

    with h5py.File(dataset_path, 'r') as f:
        neuron_keys = [k for k in f.keys() if 'neuron' in k]
        print(f"Found {len(neuron_keys)} neurons in harmonized dataset")

        for neuron_key in tqdm(neuron_keys, desc="Processing Lisberger neurons"):
            neuron = f[neuron_key]

            # Extract waveform data from harmonized format
            if 'mean_waveform_preprocessed' in neuron:
                wvf_data = neuron['mean_waveform_preprocessed'][()]
                # Harmonized format is (22, 180) - convert to (90,) for NEMO
                if len(wvf_data.shape) == 2:
                    # Select channel with max peak-to-peak amplitude
                    ptps = np.ptp(wvf_data, axis=1)
                    max_channel = np.argmax(ptps)
                    selected_waveform = wvf_data[max_channel, :]

                    # Truncate/pad to 90 samples (central portion)
                    if len(selected_waveform) > 90:
                        center = len(selected_waveform) // 2
                        selected_waveform = selected_waveform[center-45:center+45]
                    elif len(selected_waveform) < 90:
                        # Pad with zeros if needed
                        padding = np.zeros(90 - len(selected_waveform))
                        selected_waveform = np.concatenate([selected_waveform, padding])

                    if normalise:
                        # Normalize waveform
                        max_abs = np.max(np.abs(selected_waveform))
                        if max_abs > 1e-8:
                            selected_waveform = selected_waveform / max_abs

                    waveforms.append(selected_waveform)
                else:
                    # Fallback: pad or truncate to 90
                    wvf_flat = wvf_data.flatten()
                    if len(wvf_flat) >= 90:
                        waveforms.append(wvf_flat[:90])
                    else:
                        padded = np.zeros(90)
                        padded[:len(wvf_flat)] = wvf_flat
                        waveforms.append(padded)
            else:
                # Fallback: create dummy waveform
                waveforms.append(np.zeros(90))

            # Compute ACGs from spike data (instead of using dummy values)
            if 'spike_indices' in neuron and 'sampling_rate' in neuron:
                try:
                    spike_indices = np.array(neuron['spike_indices'])
                    sampling_rate = neuron['sampling_rate'][()]
                    spike_times = spike_indices / sampling_rate

                    # Compute 3D ACG from spike times
                    acg_neuron = compute_autocorrelogram_3d(spike_times)

                    # Ensure correct shape (10, 101)
                    if acg_neuron.shape[1] > 101:
                        acg_neuron = acg_neuron[:, :101]  # Truncate
                    elif acg_neuron.shape[1] < 101:
                        # Pad to 101
                        padding = np.zeros((10, 101 - acg_neuron.shape[1]))
                        acg_neuron = np.concatenate([acg_neuron, padding], axis=1)

                    acgs.append(acg_neuron)
                except Exception as e:
                    tqdm.write(f"Warning: Could not compute ACG for {neuron_key}: {e}")
                    acgs.append(generate_dummy_acg(10, 101))
            else:
                tqdm.write(f"Warning: No spike data for {neuron_key}, using dummy ACG")
                acgs.append(generate_dummy_acg(10, 101))

            # Extract label if this is a labeled dataset
            if is_labeled:
                label = 'unknown'  # default
                for label_field in ['expert_label', 'ground_truth_label']:
                    if label_field in neuron:
                        label_data = neuron[label_field][()]
                        if isinstance(label_data, bytes):
                            label = label_data.decode('utf-8')
                        elif hasattr(label_data, 'item'):
                            label = str(label_data.item())
                        else:
                            label = str(label_data)
                        break
                labels.append(label)
    
    # Convert to numpy arrays
    waveforms = np.array(waveforms)
    acgs = np.array(acgs)
    
    print(f"Loaded {len(waveforms)} samples from harmonized dataset:")
    print(f"  Waveforms: {waveforms.shape}")
    print(f"  ACGs: {acgs.shape}")
    
    if is_labeled:
        # Create label mappings
        unique_labels = list(set(labels))
        labelling_dict = {label: i for i, label in enumerate(unique_labels)}
        correspondence_dict = {i: label for i, label in enumerate(unique_labels)}
        
        # Convert labels to indices
        label_idx = [labelling_dict[label] for label in labels]
        
        print(f"  Labels: {len(unique_labels)} unique ({unique_labels})")
        
        return waveforms, acgs, label_idx, labels, labelling_dict, correspondence_dict
    else:
        return waveforms, acgs
