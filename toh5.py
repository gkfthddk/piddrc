import os
import gc
import json
import random
import h5py
import tqdm
import multiprocessing
import math
import time
import psutil
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

# --- Global Constants and Configuration ---
VERSION = "version16"
DEFAULT_STATS_PERCENTILES = (0.5, 0.9, 0.99)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_memory_usage(tag: str = "", pid: Optional[int] = None) -> None:
    """Prints the current memory usage of a process."""
    if pid is None:
        process = psutil.Process(os.getpid())
    else:
        process = psutil.Process(pid)
    mem = process.memory_info().rss / (1024 ** 2)  # RSS: 실제 메모리 사용량 (bytes → MB)
    logger.info(f"{tag} [PID {process.pid}] Memory usage: {mem:.2f} MB")

class StreamingStats:
    """Keep track of aggregate statistics for a channel."""

    def __init__(self, sample_size: int, percentiles: Sequence[float]):
        self.sample_size = sample_size
        self.percentiles = percentiles
        self.count: int = 0
        self.sum: float = 0.0
        self.sumsq: float = 0.0
        self.min_val: float = math.inf
        self.max_val: float = -math.inf
        self._sample: List[float] = []

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return

        # Promote to float64 for numerical stability.
        flat = values.astype(np.float64, copy=False).ravel()
        start_count = self.count
        self.count += flat.size
        self.sum += float(np.sum(flat))
        self.sumsq += float(np.sum(flat * flat))
        self.min_val = min(self.min_val, float(np.min(flat)))
        self.max_val = max(self.max_val, float(np.max(flat)))

        if self.sample_size <= 0:
            return

        # Fill the reservoir until it reaches the desired size.
        if len(self._sample) < self.sample_size:
            needed = self.sample_size - len(self._sample)
            take = min(needed, flat.size)
            self._sample.extend(map(float, flat[:take]))
            flat = flat[take:]
            total_seen = start_count + take
        else:
            total_seen = start_count

        if flat.size == 0:
            return

        # Reservoir sampling for the remaining values.
        for value in flat:
            total_seen += 1
            j = np.random.randint(0, total_seen)
            if j < self.sample_size:
                self._sample[j] = float(value)

    def finalize(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0, "min": math.nan, "max": math.nan,
                "mean": math.nan, "std": math.nan,
                "percentiles": {p: math.nan for p in self.percentiles},
            }

        mean = self.sum / self.count
        variance = max(self.sumsq / self.count - mean * mean, 0.0)
        std = math.sqrt(variance)

        percentile_values: Dict[float, float]
        if self.sample_size > 0 and self._sample:
            sample_array = np.asarray(self._sample, dtype=np.float64)
            percentile_values = {p: float(np.quantile(sample_array, p, method="linear")) for p in self.percentiles}
        else:
            percentile_values = {p: math.nan for p in self.percentiles}

        return {"count": int(self.count), "min": float(self.min_val), "max": float(self.max_val), "mean": float(mean), "std": float(std), "percentiles": percentile_values}

class H5FileProcessor:
    """
    A class to process and merge HDF5 files, including validation, reading,
    and writing with multiprocessing.
    """

    # Base channels that are always included
    _BASE_CHANNELS = [
        'C_amp', 'C_raw',
        'DRcalo3dHits.amplitude', 'DRcalo3dHits.amplitude_sum', 'DRcalo3dHits.cellID',
        'DRcalo3dHits.position.x', 'DRcalo3dHits.position.y', 'DRcalo3dHits.position.z',
        'DRcalo3dHits.time', 'DRcalo3dHits.time_end', 'DRcalo3dHits.type',
        'DRcalo2dHits.amplitude', 'DRcalo2dHits.cellID',
        'DRcalo2dHits.position.x', 'DRcalo2dHits.position.y', 'DRcalo2dHits.position.z',
        'DRcalo2dHits.type',
        'Reco3dHits_C.amplitude', 'Reco3dHits_C.position.x', 'Reco3dHits_C.position.y',
        'Reco3dHits_C.position.z',
        'Reco3dHits_S.amplitude', 'Reco3dHits_S.position.x', 'Reco3dHits_S.position.y',
        'Reco3dHits_S.position.z',
        'E_dep', 'E_gen', 'E_leak', 'GenParticles.PDG', 'GenParticles.momentum.phi',
        'GenParticles.momentum.theta', 'seed', 'S_amp', 'S_raw', 'angle2',
        VERSION,
    ]

    # Templates for pooled channels
    _POOL_CHANNEL_TEMPLATES = [
        "Reco3dHits{pool}_C.amplitude", "Reco3dHits{pool}_C.position.x",
        "Reco3dHits{pool}_C.position.y", "Reco3dHits{pool}_C.position.z",
        "Reco3dHits{pool}_S.amplitude", "Reco3dHits{pool}_S.position.x",
        "Reco3dHits{pool}_S.position.y", "Reco3dHits{pool}_S.position.z",
        "DRcalo3dHits{pool}.amplitude", "DRcalo3dHits{pool}.amplitude_sum",
        "DRcalo3dHits{pool}.cellID", "DRcalo3dHits{pool}.position.x",
        "DRcalo3dHits{pool}.position.y", "DRcalo3dHits{pool}.position.z",
        "DRcalo3dHits{pool}.time", "DRcalo3dHits{pool}.time_end",
        "DRcalo3dHits{pool}.type",
        "DRcalo2dHits{pool}.amplitude", "DRcalo2dHits{pool}.cellID",
        "DRcalo2dHits{pool}.position.x", "DRcalo2dHits{pool}.position.y",
        "DRcalo2dHits{pool}.position.z", "DRcalo2dHits{pool}.type",
    ]

    def __init__(
        self,
        reco_path: Path,
        output_dir: Path,
        max_entries_per_dataset: int = 3_000_000,
        compute_stats: bool = False,
        stats_sample_size: int = 200_000,
        stats_percentiles: Sequence[float] = DEFAULT_STATS_PERCENTILES,
    ):
        self.reco_path = reco_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries_per_dataset = max_entries_per_dataset
        self.write_keys = self._build_write_keys()
        self.compute_stats = compute_stats
        self.stats_kwargs = {"sample_size": stats_sample_size, "percentiles": stats_percentiles}

    def _build_write_keys(self) -> List[str]:
        """Generates the list of all dataset keys to be written."""
        keys = list(self._BASE_CHANNELS) # Start with a copy of base channels
        for pool in [4, 8, 14, 28, 56]:
            for template in self._POOL_CHANNEL_TEMPLATES:
                keys.append(template.format(pool=pool))
        return keys

    @staticmethod
    def _validated_indices(file_name: Path) -> List[int]:
        """
        Reads S_amp and C_amp from an HDF5 file and returns indices where
        both are valid (non-zero and below a threshold).
        """
        try:
            with h5py.File(file_name, 'r') as f:
                s_amp = f['S_amp'][:]
                c_amp = f['C_amp'][:]
                # Ensure s_amp and c_amp are 1D arrays for element-wise comparison
                if s_amp.ndim > 1: s_amp = s_amp.flatten()
                if c_amp.ndim > 1: c_amp = c_amp.flatten()

                # Use boolean indexing for efficiency
                valid_mask = (s_amp != 0) & (s_amp <= 2e8) & \
                             (c_amp != 0) & (c_amp <= 2e8)
                return np.where(valid_mask)[0].tolist()
        except Exception as e:
            logger.error(f"Error validating indices in {file_name}: {e}")
            return []

    @staticmethod
    def _read_datasets(file_name: Path, dataset_names: List[str], indices: List[int]) -> Tuple[Path, Optional[Dict[str, np.ndarray]]]:
        """
        Reads specified datasets from an HDF5 file for given indices.
        Returns the file name and a dictionary of data, or None if an error occurs.
        """
        data_dict = {}
        try:
            with h5py.File(file_name, 'r') as f:
                for dataset_name in dataset_names:
                    if dataset_name not in f:
                        logger.warning(f"Dataset '{dataset_name}' not found in {file_name}. Skipping.")
                        continue
                    # Use np.take for efficient indexing, especially with large datasets
                    data_dict[dataset_name] = np.take(f[dataset_name], indices, axis=0)
            return file_name, data_dict
        except Exception as e:
            logger.error(f"Failed to read datasets from {file_name}: {e}")
            return file_name, None

    @staticmethod
    def _reader_worker(file_list: List[Path], dataset_names: List[str], queue: multiprocessing.Queue):
        """Worker process to read data from HDF5 files and put it into a queue."""
        for file_name in file_list:
            indices = H5FileProcessor._validated_indices(file_name)
            if not indices:
                logger.info(f"No valid entries found or error in {file_name}. Skipping.")
                queue.put((file_name, None)) # Indicate no valid data for this file
                continue
            
            file_name, data_dict = H5FileProcessor._read_datasets(file_name, dataset_names, indices)
            queue.put((file_name, data_dict))
        logger.info(f"Reader worker finished for {len(file_list)} files.")

    def _writer_worker(self, queue: multiprocessing.Queue, output_file: Path, total_files_to_process: int, stats_dict_proxy: Dict):
        """Worker process to write data from the queue to the output HDF5 file."""
        logger.info(f"Writer worker started for {output_file}")
        processed_files_count = 0

        stats_calculators: Dict[str, StreamingStats] = {}
        if self.compute_stats:
            logger.info("Statistics computation is enabled for the writer.")
            stats_calculators = {key: StreamingStats(**self.stats_kwargs) for key in self.write_keys}

        with h5py.File(output_file, 'a') as f_out:
            with tqdm.tqdm(total=total_files_to_process, desc=f"Writing to {output_file.name}") as pbar:
                while processed_files_count < total_files_to_process:
                    file_name, data_dict = queue.get()
                    
                    if file_name is None and data_dict is None: # Sentinel value to stop the writer
                        break

                    processed_files_count += 1
                    pbar.update(1)

                    if data_dict is None:
                        logger.warning(f"Skipping writing for {file_name} due to previous read error or no valid data.")
                        continue

                    for dataset_name, data_chunk in data_dict.items():
                        # Update stats before writing
                        if self.compute_stats and dataset_name in stats_calculators:
                            stats_calculators[dataset_name].update(data_chunk)

                        if dataset_name in f_out:
                            dataset = f_out[dataset_name]
                            # Check if adding this chunk would exceed max_entries_per_dataset
                            if dataset.shape[0] + data_chunk.shape[0] > self.max_entries_per_dataset:
                                logger.warning(f"Dataset '{dataset_name}' in {output_file.name} would exceed "
                                               f"{self.max_entries_per_dataset} entries. Skipping further appends for this dataset.")
                                continue
                            dataset.resize(dataset.shape[0] + data_chunk.shape[0], axis=0)
                            dataset[-data_chunk.shape[0]:] = data_chunk
                        else:
                            maxshape = (self.max_entries_per_dataset,) + data_chunk.shape[1:]
                            f_out.create_dataset(
                                dataset_name,
                                data=data_chunk,
                                maxshape=maxshape,
                                chunks=True,
                                compression='lzf'
                            )
                    del data_dict # Free memory
                    gc.collect() # Explicit garbage collection

        if self.compute_stats:
            logger.info("Finalizing and collecting statistics...")
            for key, calculator in stats_calculators.items():
                stats_dict_proxy[key] = calculator.finalize()

        logger.info(f"Writer worker finished for {output_file}")
        print_memory_usage('Writer end')

    def _save_stats_to_h5(self, output_file: Path, stats: Dict):
        """Saves computed statistics as attributes to the HDF5 file."""
        if not self.compute_stats:
            return
        logger.info(f"Writing statistics as attributes to {output_file}")
        with h5py.File(output_file, 'a') as f: # Open in append mode to add attributes
            f.attrs['statistics'] = json.dumps(stats, allow_nan=True)
    @staticmethod
    def _check_reco_file(file_name: Path) -> Tuple[Optional[Path], Optional[int]]:
        """
        Checks if an HDF5 file exists, is readable, and contains the VERSION key.
        Returns the file path and number of entries if valid, otherwise None.
        """
        if not file_name.is_file():
            return None, None
        try:
            with h5py.File(file_name, 'r') as hf:
                keys = list(hf.keys())
                if VERSION in keys and 'seed' in keys: # Ensure 'seed' exists to get entries
                    entries = len(hf['seed'])
                    return file_name, entries
                else:
                    logger.warning(f"File {file_name} is missing '{VERSION}' or 'seed' key. Skipping.")
                    return None, None
        except Exception as e:
            logger.error(f"Error opening or reading {file_name}: {e}")
            return None, None

    def process_sample(self, sample_name: str, max_num_files: int = 0, num_reader_processes: int = 4, overwrite: bool = False):
        """
        Processes a single sample by merging multiple HDF5 files into one.
        """
        output_file = self.output_dir / f"{sample_name}.h5py"
        logger.info(f"Processing sample '{sample_name}', output to: {output_file}")

        # Initialize output file (create if not exists, clear if exists)
        if output_file.exists() and not overwrite:
            logger.warning(f"Output file {output_file} already exists and overwrite is False. Skipping.")
            return

        with h5py.File(output_file, 'w') as f:
            pass # Just creates/truncates the file

        sample_reco_dir = self.reco_path / sample_name
        if not sample_reco_dir.is_dir():
            logger.error(f"Reco directory not found: {sample_reco_dir}")
            return

        reco_list = sorted(list(sample_reco_dir.iterdir())) # Sort for consistent processing
        if max_num_files > 0:
            reco_list = reco_list[:min(max_num_files, len(reco_list))]

        logger.info(f"Found {len(reco_list)} potential reco files for '{sample_name}'.")

        valid_files: List[Path] = []
        total_entries = 0

        # Use multiprocessing Pool to check files in parallel
        with multiprocessing.Pool(processes=num_reader_processes) as pool:
            results = []
            for recopath in reco_list:
                results.append(pool.apply_async(self._check_reco_file, args=(recopath,)))

            for result in tqdm.tqdm(results, desc="Validating reco files", leave=False):
                file_name, entry_count = result.get()
                if file_name:
                    valid_files.append(file_name)
                if entry_count:
                    total_entries += entry_count

        logger.info(f"Validated {len(valid_files)} files with a total of {total_entries} entries for '{sample_name}'.")

        if not valid_files:
            logger.error(f"No valid reco files found for sample '{sample_name}'. Skipping merge.")
            return

        # Setup multiprocessing for reading and writing
        print_memory_usage('Start merge')
        manager = multiprocessing.Manager()
        data_queue = manager.Queue(maxsize=40) # Queue for data chunks
        stats_dict_proxy = manager.dict() if self.compute_stats else None

        # Start reader processes
        reader_processes: List[multiprocessing.Process] = []
        chunk_size = len(valid_files) // num_reader_processes + 1
        for i in range(num_reader_processes):
            sub_list = valid_files[i * chunk_size:(i + 1) * chunk_size]
            if not sub_list:
                continue
            p = multiprocessing.Process(
                target=self._reader_worker,
                args=(sub_list, self.write_keys, data_queue)
            )
            reader_processes.append(p)
            p.start()

        # Start writer process
        writer_process = multiprocessing.Process(
            target=self._writer_worker,
            args=(data_queue, output_file, len(valid_files), stats_dict_proxy)
        )
        writer_process.start()

        # Wait for reader processes to finish
        for p in reader_processes:
            p.join()
        logger.info("All reader processes finished.")

        # Send sentinel value to stop the writer process
        data_queue.put((None, None))
        writer_process.join()
        logger.info("Writer process finished.")

        if self.compute_stats: # stats_dict_proxy is guaranteed to be a manager.dict() if compute_stats is True
            self._save_stats_to_h5(output_file, dict(stats_dict_proxy)) # Convert Manager.dict to a regular dict

        print_memory_usage(f"Finished processing {sample_name}")
        gc.collect() # Final garbage collection

# Main execution block
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge HDF5 files and optionally compute statistics.")
    parser.add_argument("--reco_base", type=str, default="/users/yulee/dream/tools/reco", help="Set reco_base_path.")
    parser.add_argument("--ouput_dir", type=str, default="h5s", help="Set output_h5_dir.")
    parser.add_argument("--compute_stats", action="store_true", help="Enable computation of channel statistics.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    reco_base_path = Path(args.reco_base)
    output_h5_dir = Path(args.output_dir)

    processor = H5FileProcessor(reco_base_path, output_h5_dir, compute_stats=args.compute_stats)

    # Example usage:
    pids = ['pi+', 'pi0', 'gamma', 'e-']
    # pids = ['mu']
    # pids = ['e-','pi+','pi0','gamma','kaon+','proton','neutron','kaon0L']

    for pid in pids:
        # for en in ["10","20","50","100"]:
        for en in ["1-100"]:
            for pool_size in [10]: # Renamed 'pool' to 'pool_size' to avoid confusion with multiprocessing pool
                if pool_size == 1:
                    sample_name = f"{pid}_{en}GeV"
                else:
                    sample_name = f"{pid}_{en}GeV_{pool_size}"
                processor.process_sample(sample_name, max_num_files=5000, num_reader_processes=8, overwrite=args.overwrite)

                gc.collect()
                time.sleep(1)
    print_memory_usage('End of script')

if __name__ == "__main__":
    main()
