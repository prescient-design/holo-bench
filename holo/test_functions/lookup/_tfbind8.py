"""Implementation of the TFBIND8 lookup function."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
        """Load the TFBIND8 dataset.

        Returns:
            Tuple[Dict[str, float], np.ndarray, List[str]]:
                - lookup_dict: Dictionary mapping sequence strings to fitness scores
                - sorted_scores: Array of scores sorted in descending order
                - sorted_seqs: List of sequences sorted by descending score
        """
        url = "https://media.githubusercontent.com/media/francesding/variationalsearch/refs/heads/main/data/TFBIND8/tf_bind_8.csv"
        try:
            file_path = self._retrieve(url, fname="tf_bind_8.csv", cache_name="tfbind8")
            df = pd.read_csv(file_path)
            required_columns = {"sequences", "fitness"}
            if not required_columns.issubset(df.columns):
                raise ValueError(
                    f"TFBIND8 data missing required columns. Found: {df.columns}, needed: {required_columns}"
                )
            sequences = df["sequences"].tolist()
            scores = df["fitness"].values
        except ImportError:
            raise
        except Exception:
            sequences, scores = self._load_flexs_fallback()

        return self._build_lookup(sequences, scores)

    def _load_flexs_fallback(self) -> Tuple[List[str], np.ndarray]:
        """Fall back to the FLEXS TF-binding dataset for ``self.transcription_factor``."""
        url = f"https://raw.githubusercontent.com/samsinai/FLEXS/master/flexs/landscapes/data/tf_binding/{self.transcription_factor}_8mers.txt"
        file_path = self._retrieve(url, fname=f"{self.transcription_factor}_8mers.txt", cache_name="tfbind8")
        df = pd.read_csv(file_path, sep="\t")

        sequences_1 = df["8-mer"].tolist()
        sequences_2 = df["8-mer.1"].tolist() if "8-mer.1" in df.columns else []
        sequences = sequences_1 + sequences_2

        e_scores = df["E-score"].values
        normalized = (e_scores - e_scores.min()) / (e_scores.max() - e_scores.min())
        scores = np.concatenate([normalized, normalized[: len(sequences_2)]]) if sequences_2 else normalized
        return sequences, scores
