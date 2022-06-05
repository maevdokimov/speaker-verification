from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.nn.functional import normalize


class EmbeddingStorage:
    """
    Encapsulates embeddings, computed for audios
    Args:
        device: where to store embeddings
        name_to_embed: dict with mapping (file_path -> embedding)
        normalized: whether to store raw or normalized embeddings
    """

    def __init__(
        self, device: torch.device, name_to_embed: Optional[Dict[str, torch.Tensor]] = None, normalized: bool = True
    ):
        self.device = device
        self.name_to_embed = name_to_embed or {}
        self.normalized = normalized

    def save_embeddings(self, output_path: Path):
        """
        Dumps embeddings to the disk
        """
        _name_to_embed = {k: v.cpu() for k, v in self.name_to_embed.items()}
        _output_dict = {"name_to_embed": _name_to_embed, "normalized": self.normalized}

        torch.save(_output_dict, output_path)

    @classmethod
    def load_embeddings(cls, input_path: Path, device: torch.device):
        _input_dict = torch.load(input_path, map_location=device)

        return cls(device, _input_dict["name_to_embed"], _input_dict["normalized"])

    def __setitem__(self, path: Union[Path, str], embedding: torch.Tensor):
        """
        Args:
            path: path to input file. Paths are processed to match input of voxceleb
                test files: .../.../file.wav
            embedding: embedding to store
        """
        if isinstance(path, str):
            path = Path(path)

        resolved_path = path.resolve()
        processed_path = str(resolved_path.relative_to(resolved_path.parents[2]))

        if self.normalized:
            embedding = normalize(embedding)
        self.name_to_embed[processed_path] = embedding.to(self.device)

    def __getitem__(self, path: str):
        """
        Args:
            path: path in format .../.../file.wav
        """
        return self.name_to_embed[path]
