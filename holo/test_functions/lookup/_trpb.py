"""TRPB: TrpB 4-residue tryptophan synthase optimization benchmark.

This test function is a lookup table for the TrpB dataset, a 4-residue amino acid
sequence optimization task for tryptophan synthase. The data comes from Johnston et al. 2024.

The sequence space is 20^4 = 160,000 possible sequences (of which 153,620 are
measured in the dataset). The benchmark task is to find sequences with high fitness
(enzyme activity).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from holo.test_functions.lookup._abstract_lookup import AbstractLookup


class TRPBLookup(AbstractLookup):
    """TrpB 4-residue tryptophan synthase optimization benchmark.

    This benchmark represents the fitness landscape for a 4-residue sequence
    optimization task for tryptophan synthase beta enzyme. The fitness values
    represent enzyme activity, with higher values being better.

    The sequence space consists of 4 amino acid positions using the standard 20-letter
    amino acid alphabet, resulting in 20^4 = 160,000 possible sequences.
    The dataset contains measurements for 153,620 of these sequences.

    Data source: Johnston et al., 2024. https://doi.org/10.22002/h5rah-5z170
    "Mapping protein sequence-function relationships using multiple large-scale
    inverse folding models"
    """

    def __init__(
        self,
        dim: int = 4,
        noise_std: Optional[float] = None,
        negate: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize TrpB lookup function.

        Args:
            dim: The dimension (sequence length). Must be 4 for this benchmark.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function value.
            device: The torch device to use.
        """
        if dim != 4:
            raise ValueError(
                f"TrpBLookup only supports dim=4 (got {dim}). "
                f"This benchmark uses a fixed 4-residue amino acid sequence."
            )

        # Initialize the base class
        super().__init__(
            dim=dim,
            alphabet="ARNDCQEGHILKMFPSTWYV",  # Standard 20 amino acids
            wildtype_sequence="VFVS",  # The wild-type sequence from the paper
            noise_std=noise_std,
            negate=negate,
            device=device,
        )

    def _load_data(self) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
        """Load the TrpB dataset from the variationalsearch repository.

        Returns:
            A tuple containing:
                - lookup_dict: Dictionary mapping sequence strings to fitness scores
                - sorted_scores: Array of scores sorted in descending order
                - sorted_seqs: List of sequences sorted by descending score
        """
        url = "https://media.githubusercontent.com/media/francesding/variationalsearch/refs/heads/main/data/TRPB/four-site_simplified_AA_data.csv"
        file_path = self._retrieve(url, fname="four-site_simplified_AA_data.csv", cache_name="trpb")
        df = pd.read_csv(file_path)

        required_columns = {"AAs", "fitness"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"TrpB data missing required columns. Found: {df.columns}, needed: {required_columns}")

        if "# Stop" in df.columns:
            df = df[df["# Stop"] < 1]

        sequences = df["AAs"].tolist()
        scores = df["fitness"].values
        return self._build_lookup(sequences, scores)
