#!/usr/bin/env python3
"""
Allen NWB Data Loader for NEMO Benchmark

Provides utilities to load spike times, waveforms, and metadata from
Allen Brain Observatory NWB files. Supports both local NWB files and
remote S3 access (when S3_BUCKET / S3_PREFIX / S3_ENDPOINT are set).

Usage:
    # Local NWB file (default workflow — no S3 required)
    loader = AllenNWBLoader("/path/to/session.nwb")
    spike_times = loader.get_spike_times()
    waveforms = loader.get_waveforms()

    # Optional S3 access (requires AWS credentials and S3_* env vars)
    loader = AllenNWBLoader.from_s3("session_id.nwb")
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Try to import required packages
try:
    from pynwb import NWBHDF5IO
    PYNWB_AVAILABLE = True
except ImportError:
    PYNWB_AVAILABLE = False
    warnings.warn("PyNWB not available. Install with: pip install pynwb")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    warnings.warn("h5py not available. Install with: pip install h5py")


# S3 configuration for optional remote access. Override via environment
# variables; when loading .nwb files from disk, none of these are needed.
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "")


class AllenNWBLoader:
    """
    Load data from Allen Brain Observatory NWB files.
    """

    def __init__(self, nwb_path: str, verbose: bool = True):
        """
        Initialize the loader with an NWB file path.

        Args:
            nwb_path: Path to the NWB file
            verbose: Whether to print progress messages
        """
        self.nwb_path = Path(nwb_path)
        self.verbose = verbose

        if not self.nwb_path.exists():
            raise FileNotFoundError(f"NWB file not found: {nwb_path}")

        self._nwbfile = None
        self._io = None
        self._units_df = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open the NWB file for reading."""
        if not PYNWB_AVAILABLE:
            raise ImportError("PyNWB is required. Install with: pip install pynwb")

        if self._io is None:
            self._io = NWBHDF5IO(str(self.nwb_path), 'r')
            self._nwbfile = self._io.read()
            if self.verbose:
                print(f"Opened NWB file: {self.nwb_path.name}")

    def close(self):
        """Close the NWB file."""
        if self._io is not None:
            self._io.close()
            self._io = None
            self._nwbfile = None

    @property
    def nwbfile(self):
        """Get the NWB file object, opening if necessary."""
        if self._nwbfile is None:
            self.open()
        return self._nwbfile

    def get_session_info(self) -> Dict:
        """Get session metadata."""
        nwbfile = self.nwbfile
        return {
            'session_id': nwbfile.session_id if hasattr(nwbfile, 'session_id') else None,
            'session_description': nwbfile.session_description,
            'session_start_time': str(nwbfile.session_start_time),
            'experimenter': nwbfile.experimenter if hasattr(nwbfile, 'experimenter') else None,
            'institution': nwbfile.institution if hasattr(nwbfile, 'institution') else None,
        }

    def get_unit_ids(self) -> np.ndarray:
        """Get array of unit IDs."""
        nwbfile = self.nwbfile
        if nwbfile.units is None:
            return np.array([])
        return np.array(nwbfile.units.id[:])

    def get_num_units(self) -> int:
        """Get the number of units."""
        nwbfile = self.nwbfile
        if nwbfile.units is None:
            return 0
        return len(nwbfile.units)

    def get_spike_times(self, unit_id: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        Get spike times for units.

        Args:
            unit_id: Optional specific unit ID. If None, returns all units.

        Returns:
            Dictionary mapping unit_id -> spike_times (in seconds)
        """
        nwbfile = self.nwbfile
        if nwbfile.units is None:
            return {}

        spike_times_dict = {}
        units = nwbfile.units

        if unit_id is not None:
            # Get single unit
            idx = np.where(units.id[:] == unit_id)[0]
            if len(idx) > 0:
                spike_times_dict[unit_id] = units['spike_times'][idx[0]]
        else:
            # Get all units
            for i, uid in enumerate(units.id[:]):
                spike_times_dict[uid] = units['spike_times'][i]

        return spike_times_dict

    def get_spike_times_samples(self,
                                 unit_id: Optional[int] = None,
                                 sampling_rate: float = 30000.0) -> Dict[int, np.ndarray]:
        """
        Get spike times in samples.

        Args:
            unit_id: Optional specific unit ID
            sampling_rate: Sampling rate to convert to samples

        Returns:
            Dictionary mapping unit_id -> spike_times (in samples)
        """
        spike_times_s = self.get_spike_times(unit_id)
        return {
            uid: np.round(times * sampling_rate).astype(np.int64)
            for uid, times in spike_times_s.items()
        }

    def get_waveforms(self, unit_id: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        Get mean waveforms for units.

        Args:
            unit_id: Optional specific unit ID

        Returns:
            Dictionary mapping unit_id -> waveform array
        """
        nwbfile = self.nwbfile
        if nwbfile.units is None:
            return {}

        waveforms_dict = {}
        units = nwbfile.units

        # Check if waveforms are stored
        if 'waveform_mean' not in units.colnames:
            if self.verbose:
                print("WARNING: No waveform_mean column in units table")
            return {}

        if unit_id is not None:
            idx = np.where(units.id[:] == unit_id)[0]
            if len(idx) > 0:
                waveforms_dict[unit_id] = units['waveform_mean'][idx[0]]
        else:
            for i, uid in enumerate(units.id[:]):
                waveforms_dict[uid] = units['waveform_mean'][i]

        return waveforms_dict

    def get_unit_locations(self) -> Dict[int, Dict]:
        """
        Get brain region/location for each unit.

        Returns:
            Dictionary mapping unit_id -> location info dict
        """
        nwbfile = self.nwbfile
        if nwbfile.units is None:
            return {}

        locations = {}
        units = nwbfile.units

        location_cols = ['location', 'brain_area', 'ccf_structure', 'ecephys_structure_acronym']

        for i, uid in enumerate(units.id[:]):
            loc_info = {}
            for col in location_cols:
                if col in units.colnames:
                    val = units[col][i]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    loc_info[col] = val
            locations[uid] = loc_info

        return locations

    def get_unit_quality(self) -> Dict[int, str]:
        """
        Get quality labels for units (e.g., 'good', 'mua').

        Returns:
            Dictionary mapping unit_id -> quality label
        """
        nwbfile = self.nwbfile
        if nwbfile.units is None:
            return {}

        quality = {}
        units = nwbfile.units

        quality_cols = ['quality', 'isi_violations', 'snr']

        for i, uid in enumerate(units.id[:]):
            if 'quality' in units.colnames:
                val = units['quality'][i]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                quality[uid] = val
            else:
                quality[uid] = 'unknown'

        return quality

    def to_nemo_format(self,
                       target_waveform_length: int = 90,
                       sampling_rate: float = 30000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert NWB data to NEMO format.

        Args:
            target_waveform_length: Target length for waveforms
            sampling_rate: Sampling rate for spike time conversion

        Returns:
            Tuple of (waveforms, spike_times_dict, unit_ids)
            - waveforms: (N, target_waveform_length) array
            - spike_times_dict: dict mapping unit_id -> spike times in samples
            - unit_ids: array of unit IDs
        """
        unit_ids = self.get_unit_ids()
        spike_times = self.get_spike_times_samples(sampling_rate=sampling_rate)
        waveforms_raw = self.get_waveforms()

        # Process waveforms to target length
        waveforms_list = []
        valid_unit_ids = []

        for uid in unit_ids:
            if uid not in waveforms_raw:
                continue

            wvf = waveforms_raw[uid]

            # Handle multi-channel waveforms - select peak channel
            if wvf.ndim == 2:
                ptps = np.ptp(wvf, axis=1)
                peak_ch = np.argmax(ptps)
                wvf = wvf[peak_ch, :]

            # Truncate or pad to target length
            if len(wvf) > target_waveform_length:
                # Take central portion
                center = len(wvf) // 2
                half = target_waveform_length // 2
                wvf = wvf[center - half:center + half + target_waveform_length % 2]
            elif len(wvf) < target_waveform_length:
                # Pad with zeros
                pad = target_waveform_length - len(wvf)
                wvf = np.pad(wvf, (pad // 2, pad - pad // 2))

            # Normalize
            max_abs = np.max(np.abs(wvf))
            if max_abs > 1e-8:
                wvf = wvf / max_abs

            waveforms_list.append(wvf)
            valid_unit_ids.append(uid)

        waveforms = np.stack(waveforms_list)
        valid_unit_ids = np.array(valid_unit_ids)

        # Filter spike times to valid units
        spike_times_filtered = {
            uid: spike_times.get(uid, np.array([]))
            for uid in valid_unit_ids
        }

        return waveforms, spike_times_filtered, valid_unit_ids

    @classmethod
    def from_s3(cls,
                filename: str,
                local_cache_dir: Optional[str] = None,
                verbose: bool = True) -> 'AllenNWBLoader':
        """
        Load an NWB file from S3.

        Args:
            filename: Name of the NWB file in S3
            local_cache_dir: Directory to cache downloaded files
            verbose: Whether to print progress messages

        Returns:
            AllenNWBLoader instance
        """
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError("boto3 is required for S3 access. Install with: pip install boto3")

        if local_cache_dir is None:
            local_cache_dir = os.path.join(os.path.expanduser("~"), ".allen_nwb_cache")

        os.makedirs(local_cache_dir, exist_ok=True)
        local_path = os.path.join(local_cache_dir, filename)

        # Download if not cached
        if not os.path.exists(local_path):
            if verbose:
                print(f"Downloading {filename} from S3...")

            s3_client = boto3.client(
                's3',
                endpoint_url=S3_ENDPOINT,
                config=Config(signature_version='s3v4')
            )

            s3_key = S3_PREFIX + filename
            s3_client.download_file(S3_BUCKET, s3_key, local_path)

            if verbose:
                print(f"Downloaded to: {local_path}")

        return cls(local_path, verbose=verbose)


def list_allen_nwb_files_s3() -> List[str]:
    """
    List available Allen NWB files in S3.

    Returns:
        List of NWB filenames
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError("boto3 is required for S3 access")

    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        config=Config(signature_version='s3v4')
    )

    response = s3_client.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX
    )

    files = []
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.nwb'):
                filename = key[len(S3_PREFIX):]
                files.append(filename)

    return files


def load_allen_data_for_nemo(dataset_path: str,
                              target_waveform_length: int = 90) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Load Allen dataset for NEMO from a directory containing NWB files.

    Args:
        dataset_path: Path to directory with NWB files
        target_waveform_length: Target waveform length

    Returns:
        Tuple of (waveforms, spike_times_dict, unit_ids)
    """
    dataset_path = Path(dataset_path)

    # Find NWB files
    nwb_files = list(dataset_path.glob("*.nwb"))

    if not nwb_files:
        raise FileNotFoundError(f"No NWB files found in {dataset_path}")

    all_waveforms = []
    all_spike_times = {}
    all_unit_ids = []

    for nwb_file in nwb_files:
        print(f"Loading {nwb_file.name}...")

        with AllenNWBLoader(str(nwb_file)) as loader:
            waveforms, spike_times, unit_ids = loader.to_nemo_format(
                target_waveform_length=target_waveform_length
            )

            # Prefix unit IDs with session name to ensure uniqueness
            session_prefix = nwb_file.stem
            prefixed_ids = [f"{session_prefix}_{uid}" for uid in unit_ids]

            all_waveforms.append(waveforms)
            all_unit_ids.extend(prefixed_ids)

            for uid, prefix_uid in zip(unit_ids, prefixed_ids):
                all_spike_times[prefix_uid] = spike_times[uid]

    # Stack waveforms
    all_waveforms = np.vstack(all_waveforms)
    all_unit_ids = np.array(all_unit_ids)

    print(f"Loaded {len(all_unit_ids)} units from {len(nwb_files)} NWB files")

    return all_waveforms, all_spike_times, all_unit_ids


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Load Allen NWB data")
    parser.add_argument("nwb_path", help="Path to NWB file or directory")
    parser.add_argument("--output", help="Output directory for converted data")
    parser.add_argument("--list-s3", action="store_true", help="List available S3 files")

    args = parser.parse_args()

    if args.list_s3:
        print("Available Allen NWB files in S3:")
        for f in list_allen_nwb_files_s3():
            print(f"  {f}")
    else:
        path = Path(args.nwb_path)

        if path.is_dir():
            waveforms, spike_times, unit_ids = load_allen_data_for_nemo(str(path))
        else:
            with AllenNWBLoader(str(path)) as loader:
                waveforms, spike_times, unit_ids = loader.to_nemo_format()

        print(f"Waveforms shape: {waveforms.shape}")
        print(f"Number of units: {len(unit_ids)}")
        print(f"Example unit IDs: {unit_ids[:5]}")

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            np.save(os.path.join(args.output, "waveforms.npy"), waveforms)
            np.save(os.path.join(args.output, "unit_ids.npy"), unit_ids)
            print(f"Saved to {args.output}")
