"""DHFR: DHFR 9-mer dihydrofolate reductase optimization benchmark.

This test function is a lookup table for the DHFR dataset, a 9-mer DNA sequence
optimization task for dihydrofolate reductase. The data comes from Papkou et al., 2023.

The sequence space is 4^9 = 262,144 possible sequences. The benchmark task is to
find sequences with high fitness.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
        """Load the DHFR dataset from the variationalsearch repository.

        Returns:
            A tuple containing:
                - lookup_dict: Dictionary mapping sequence strings to fitness scores
                - sorted_scores: Array of scores sorted in descending order
                - sorted_seqs: List of sequences sorted by descending score
        """
        url = "https://media.githubusercontent.com/media/francesding/variationalsearch/refs/heads/main/data/DHFR/DHFR_fitness_data_wt.csv"
        file_path = self._retrieve(url, fname="DHFR_fitness_data_wt.csv", cache_name="dhfr")
        df = pd.read_csv(file_path, index_col=0)

        required_columns = {"SV", "m"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DHFR data missing required columns. Found: {df.columns}, needed: {required_columns}")

        sequences = df["SV"].tolist()
        scores = df["m"].values
        # First sequence in this dataset is the wildtype.
        self.wildtype_sequence = sequences[0]
        return self._build_lookup(sequences, scores)
