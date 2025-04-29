"""Implementation of the TFBIND8 lookup function."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pooch
import torch

from holo.test_functions.lookup._abstract_lookup import AbstractLookup


class TFBIND8Lookup(AbstractLookup):
    """
    TFBIND8 Lookup function.

    This is a lookup-based function for the 8-mer DNA transcription factor binding
    dataset. The dataset contains 65,792 sequences of 8 nucleotides and their
    corresponding binding affinity scores.

    The DNA alphabet is {A, C, G, T} encoded as integers {0, 1, 2, 3}.

    Args:
        noise_std: float, default=0.0
            Standard deviation of Gaussian noise added to the output.
        negate: bool, default=False
            If True, negate the function values. Default is maximizing the binding affinity.
        dim: int, default=8
            Sequence length. Must be 8 for this benchmark.
        transcription_factor: str, default="SIX6_REF_R1"
            The transcription factor dataset to use.
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        negate: bool = False,
        dim: int = 8,  # Added for config compatibility, but can't be changed
        transcription_factor: str = "SIX6_REF_R1",  # Default TF
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the TFBIND8 lookup function."""
        if dim != 8:
            raise ValueError("TFBIND8Lookup has a fixed sequence length of 8 and cannot be changed.")

        self.transcription_factor = transcription_factor

        # Initialize the base class
        super().__init__(
            dim=dim,
            alphabet="ACGT",  # DNA alphabet
            wildtype_sequence=None,  # Will be determined from data if available
            noise_std=noise_std,
            negate=negate,
            device=device,
        )

    def _load_data(self) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
        """
        Load the TFBIND8 dataset using pooch.

        Returns:
            Tuple[Dict[str, float], np.ndarray, List[str]]:
                - lookup_dict: Dictionary mapping sequence strings to fitness scores
                - sorted_scores: Array of scores sorted in descending order
                - sorted_seqs: List of sequences sorted by descending score
        """
        # Remote URL for the data
        url = "https://media.githubusercontent.com/media/francesding/variationalsearch/refs/heads/main/data/TFBIND8/tf_bind_8.csv"

        # Use pooch to download and cache the data
        try:
            file_path = pooch.retrieve(
                url,
                known_hash=None,  # We're not checking the hash for now
                fname="tf_bind_8.csv",
                path=pooch.os_cache("tfbind8"),
            )

            # Read the data file
            df = pd.read_csv(file_path)

            # Make sure the data has the expected columns
            required_columns = {"sequences", "fitness"}
            if not required_columns.issubset(df.columns):
                raise ValueError(
                    f"TFBIND8 data missing required columns. Found: {df.columns}, needed: {required_columns}"
                )

            # Extract sequences and scores
            sequences = df["sequences"].tolist()
            scores = df["fitness"].values

        except Exception:
            # If there's an error with the CSV, fallback to the TF binding dataset from FLEXS
            url = f"https://raw.githubusercontent.com/samsinai/FLEXS/master/flexs/landscapes/data/tf_binding/{self.transcription_factor}_8mers.txt"

            # Use pooch to download and cache the data
            file_path = pooch.retrieve(
                url,
                known_hash=None,  # We're not checking the hash for now
                fname=f"{self.transcription_factor}_8mers.txt",
                path=pooch.os_cache("tfbind8"),
            )

            # Read the data file
            df = pd.read_csv(file_path, sep="\t")

            # Extract sequences and scores
            sequences_1 = df["8-mer"].tolist()
            sequences_2 = df["8-mer.1"].tolist() if "8-mer.1" in df.columns else []
            all_sequences = sequences_1 + sequences_2

            # Extract E-scores and normalize them to [0, 1]
            e_scores = df["E-score"].values
            # Normalize scores to [0, 1]
            normalized_scores = (e_scores - e_scores.min()) / (e_scores.max() - e_scores.min())
            all_scores = np.concatenate(
                [normalized_scores, normalized_scores[: len(sequences_2)]] if sequences_2 else [normalized_scores]
            )

            sequences = all_sequences
            scores = all_scores

        # Create lookup dictionary
        lookup_dict = {seq: score for seq, score in zip(sequences, scores)}

        # Sort sequences by score
        seqs = list(lookup_dict.keys())
        scores = np.array([lookup_dict[seq] for seq in seqs])
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        sorted_scores = scores[sorted_indices]
        sorted_seqs = [seqs[i] for i in sorted_indices]

        return lookup_dict, sorted_scores, sorted_seqs
