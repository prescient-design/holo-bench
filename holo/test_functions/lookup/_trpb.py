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
import pooch
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
        """Load the TrpB dataset from variationalsearch repository using pooch.

        Returns:
            A tuple containing:
                - lookup_dict: Dictionary mapping sequence strings to fitness scores
                - sorted_scores: Array of scores sorted in descending order
                - sorted_seqs: List of sequences sorted by descending score
        """
        # Remote URL for the data
        url = "https://raw.githubusercontent.com/skalyaanamoorthy/variationalsearch/main/data/TRPB/four-site_simplified_AA_data.csv"

        # Use pooch to download and cache the data
        file_path = pooch.retrieve(
            url,
            known_hash=None,  # We're not checking the hash for now
            fname="four-site_simplified_AA_data.csv",
            path=pooch.os_cache("trpb"),
        )

        # Read the data file
        df = pd.read_csv(file_path)

        # Make sure the data has the expected columns
        required_columns = {"AAs", "fitness"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"TrpB data missing required columns. Found: {df.columns}, needed: {required_columns}")

        # Remove rows with stop codons if that column exists
        if "# Stop" in df.columns:
            df = df[df["# Stop"] < 1]

        # Extract sequences and scores
        sequences = df["AAs"].tolist()
        scores = df["fitness"].values

        # Create lookup dictionary
        lookup_dict = {seq: score for seq, score in zip(sequences, scores)}

        # Sort sequences by score
        seqs = list(lookup_dict.keys())
        scores = np.array([lookup_dict[seq] for seq in seqs])
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        sorted_scores = scores[sorted_indices]
        sorted_seqs = [seqs[i] for i in sorted_indices]

        return lookup_dict, sorted_scores, sorted_seqs
