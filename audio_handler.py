from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import soundfile as sf


class AudioHandler:
    """Utility class to read audio files and expose commonly needed metadata."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._audio_data: Optional[np.ndarray] = None
        self._left_channel: Optional[np.ndarray] = None
        self._right_channel: Optional[np.ndarray] = None
        self._sampling_rate: Optional[int] = None
        self._audio_length: Optional[float] = None
        self._audio_metadata: Dict[str, Optional[str | int | float]] = {}
        self._load_audio()

    def _load_audio(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")

        self._audio_metadata = self._extract_metadata()

        # Avoid forcing mono so we can detect individual channels later.
        audio_data, sampling_rate = librosa.load(self.file_path, sr=None, mono=False)
        self._audio_data = audio_data
        self._sampling_rate = sampling_rate

        if audio_data.ndim == 1:
            self._left_channel = audio_data
            self._right_channel = None
        else:
            self._left_channel = audio_data[0]
            self._right_channel = audio_data[1]

        if sampling_rate > 0:
            samples = self._left_channel.shape[-1] if self._left_channel is not None else 0
            self._audio_length = samples / sampling_rate
        else:
            self._audio_length = None

        # Replace channel count with what the decoded array actually contains.
        inferred_channels = 1 if audio_data.ndim == 1 else audio_data.shape[0]
        self._audio_metadata["number_of_channels"] = inferred_channels

    def _extract_metadata(self) -> Dict[str, Optional[str | int | float]]:
        try:
            info = sf.info(str(self.file_path))
        except RuntimeError as exc:
            raise ValueError(f"Unable to read audio metadata: {self.file_path}") from exc

        bits_per_sample = self._bits_from_subtype(info.subtype)
        byte_rate = None
        block_alignment = None

        if bits_per_sample is not None and info.channels and info.samplerate:
            bytes_per_sample = bits_per_sample / 8
            block_alignment = info.channels * bytes_per_sample
            byte_rate = info.samplerate * block_alignment

        return {
            "format": info.format,
            "format_info": info.format_info,
            "subtype": info.subtype,
            "subtype_info": info.subtype_info,
            "number_of_channels": info.channels,
            "sample_rate": info.samplerate,
            "bits_per_sample": bits_per_sample,
            "byte_rate": byte_rate,
            "block_alignment": block_alignment,
            "sample_format_detail": self._describe_sample_format(info.subtype, info.subtype_info),
        }

    @staticmethod
    def _bits_from_subtype(subtype: Optional[str]) -> Optional[int]:
        if not subtype:
            return None

        subtype_map = {
            "PCM_S8": 8,
            "PCM_U8": 8,
            "PCM_16": 16,
            "PCM_24": 24,
            "PCM_32": 32,
            "PCM_64": 64,
            "FLOAT": 32,
            "DOUBLE": 64,
            "ULAW": 8,
            "ALAW": 8,
        }
        return subtype_map.get(subtype)

    @staticmethod
    def _describe_sample_format(subtype: Optional[str], subtype_info: Optional[str]) -> Optional[str]:
        if not subtype and not subtype_info:
            return None

        subtype_upper = (subtype or "").upper()
        integer_subtypes = {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32", "PCM_64"}
        float_subtypes = {"FLOAT", "DOUBLE"}

        if subtype_upper in integer_subtypes:
            base_desc = "integer"
        elif subtype_upper in float_subtypes:
            base_desc = "float"
        else:
            base_desc = "unknown"

        detail = subtype_info or subtype or "Unknown subtype"
        return f"{detail} ({base_desc})"

    @property
    def audio_data(self) -> np.ndarray:
        if self._audio_data is None:
            raise ValueError("Audio data has not been loaded.")
        return self._left_channel if self._left_channel is not None else self._audio_data

    @property
    def left_channel(self) -> np.ndarray:
        if self._left_channel is None:
            raise ValueError("Left channel data is unavailable.")
        return self._left_channel

    @property
    def right_channel(self) -> Optional[np.ndarray]:
        return self._right_channel if self._right_channel is not None else self._audio_data

    @property
    def sampling_rate(self) -> int:
        if self._sampling_rate is None:
            raise ValueError("Sampling rate unavailable.")
        return self._sampling_rate

    @property
    def audio_length(self) -> Optional[float]:
        return self._audio_length

    @property
    def sample_count(self) -> Optional[int]:
        return len(self.left_channel)

    @property
    def is_stereo(self) -> bool:
        return self._right_channel is not None

    @property
    def channel_count(self) -> Optional[int]:
        return self._audio_metadata.get("number_of_channels") if self._audio_metadata else None

    @property
    def audio_format(self) -> Optional[str]:
        return self._audio_metadata.get("format") if self._audio_metadata else None

    @property
    def bits_per_sample(self) -> Optional[int]:
        return self._audio_metadata.get("bits_per_sample") if self._audio_metadata else None

    @property
    def byte_rate(self) -> Optional[float]:
        return self._audio_metadata.get("byte_rate") if self._audio_metadata else None

    @property
    def block_alignment(self) -> Optional[float]:
        return self._audio_metadata.get("block_alignment") if self._audio_metadata else None

    @property
    def sample_format_detail(self) -> Optional[str]:
        return self._audio_metadata.get("sample_format_detail") if self._audio_metadata else None

    def get_audio_metadata(self) -> Dict[str, Optional[str | int | float]]:
        if not self._audio_metadata:
            raise ValueError("Audio metadata unavailable.")
        metadata = self._audio_metadata.copy()
        metadata.setdefault("audio_length", self._audio_length)
        metadata.setdefault("is_stereo", self.is_stereo)
        return metadata

    def print_audio_info(self) -> None:
        metadata = self.get_audio_metadata()
        printable = {
            "Audio format": metadata.get("format_info") or metadata.get("format"),
            "Number of channels": metadata.get("number_of_channels"),
            "Sample rate": metadata.get("sample_rate"),
            "Bits per sample": metadata.get("bits_per_sample"),
            "Byte rate": metadata.get("byte_rate"),
            "Block alignment": metadata.get("block_alignment"),
            "Sample format detail": metadata.get("sample_format_detail"),
            "Duration (s)": metadata.get("audio_length"),
        }

        for label, value in printable.items():
            print(f"{label}: {value if value is not None else 'Unknown'}")
