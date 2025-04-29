"""DHFR: DHFR 9-mer dihydrofolate reductase optimization benchmark.

This test function is a lookup table for the DHFR dataset, a 9-mer DNA sequence
optimization task for dihydrofolate reductase. The data comes from Papkou et al., 2023.

The sequence space is 4^9 = 262,144 possible sequences. The benchmark task is to
find sequences with high fitness.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pooch
import torch

from holo.test_functions.lookup._abstract_lookup import AbstractLookup


class DHFRLookup(AbstractLookup):
    """DHFR 9-mer dihydrofolate reductase optimization benchmark.

    This benchmark represents the fitness landscape for a 9-nucleotide sequence
    optimization task for dihydrofolate reductase enzyme. The fitness values
    represent enzyme activity, with higher values being better.

    The sequence space consists of 9 nucleotide positions using the standard DNA
    alphabet (A, C, G, T), resulting in 4^9 = 262,144 possible sequences.
    The dataset contains measurements for a large subset of these sequences.

    Data source: Papkou et al., 2023.
    """

    def __init__(
        self,
        dim: int = 9,
        noise_std: Optional[float] = None,
        negate: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize DHFR lookup function.

        Args:
            dim: The dimension (sequence length). Must be 9 for this benchmark.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function value.
            device: The torch device to use.
        """
        if dim != 9:
            raise ValueError(
                f"DHFRLookup only supports dim=9 (got {dim}). " f"This benchmark uses a fixed 9-nucleotide sequence."
            )

        # Initialize with a placeholder wildtype sequence, which will be updated after loading data
        super().__init__(
            dim=dim,
            alphabet="ACGT",  # Standard DNA alphabet
            wildtype_sequence="ATGGTTGAT",  # Placeholder - will be updated after loading data
            noise_std=noise_std,
            negate=negate,
            device=device,
        )

    def _load_data(self) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
        """Load the DHFR dataset from variationalsearch repository using pooch.

        Returns:
            A tuple containing:
                - lookup_dict: Dictionary mapping sequence strings to fitness scores
                - sorted_scores: Array of scores sorted in descending order
                - sorted_seqs: List of sequences sorted by descending score
        """
        # Remote URL for the data
        url = "https://media.githubusercontent.com/media/francesding/variationalsearch/refs/heads/main/data/DHFR/DHFR_fitness_data_wt.csv"

        # Use pooch to download and cache the data
        file_path = pooch.retrieve(
            url,
            known_hash=None,  # We're not checking the hash for now
            fname="DHFR_fitness_data_wt.csv",
            path=pooch.os_cache("dhfr"),
        )

        # Read the data file
        df = pd.read_csv(file_path, index_col=0)

        # Make sure the data has the expected columns
        required_columns = {"SV", "m"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DHFR data missing required columns. Found: {df.columns}, needed: {required_columns}")

        # Extract sequences and scores
        sequences = df["SV"].tolist()
        scores = df["m"].values  # 'm' is the target column for DHFR

        # Update wildtype sequence based on actual data
        self.wildtype_sequence = sequences[0]  # First sequence is the wildtype in this dataset

        # Create lookup dictionary
        lookup_dict = {seq: score for seq, score in zip(sequences, scores)}

        # Sort sequences by score
        seqs = list(lookup_dict.keys())
        scores = np.array([lookup_dict[seq] for seq in seqs])
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        sorted_scores = scores[sorted_indices]
        sorted_seqs = [seqs[i] for i in sorted_indices]

        return lookup_dict, sorted_scores, sorted_seqs
