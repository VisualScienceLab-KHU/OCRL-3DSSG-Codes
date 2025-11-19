import os, glob, pickle
from pathlib import Path
from typing import List, Tuple, Dict, Sequence, Any
from config import dataset_config, config_system, PREPROCESS_PATH

import torch
from torch.utils.data import Dataset

class FeatPerLabelDataset(Dataset):
    """
    여러 pickle 파일로 분산된 {label_text: feature_vectors} 구조를
    한데 모아 (feature, class_idx) 튜플을 반환하는 PyTorch Dataset.

    Parameters
    ----------
    pkl_dir : str | Path
        feat_per_labels_*.pkl 파일들이 들어 있는 디렉터리
    classes_txt : str | Path
        클래스 이름이 줄 단위로 정렬된 텍스트 파일
    in_memory : bool, default=True
        True  → 모든 feature를 메모리로 로드(가장 간단·빠름)
        False → 파일‑/오프셋 인덱스를 기록해 필요할 때 로드(메모리 절약)
    dtype : torch.dtype
        feature 벡터를 변환할 자료형(torch.float32 권장)
    """

    def __init__(
        self,
        pkl_dir: str,
        in_memory: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.pkl_dir  = Path(pkl_dir)
        self.dtype    = dtype
        self.in_memory = in_memory

        # ──────────────────────────────────────────────
        # 1) 클래스 텍스트 → 정수 idx 매핑
        # ──────────────────────────────────────────────
        with open(f"{dataset_config['root']}/classes.txt", "r", encoding="utf-8") as f:
            self.class_texts: List[str] = [ln.strip() for ln in f if ln.strip()]
        self.text2idx: Dict[str, int] = {
            txt: i for i, txt in enumerate(self.class_texts)
        }

        # ──────────────────────────────────────────────
        # 2) pickle 파일 목록 수집
        #    (glob 패턴은 필요에 맞게 수정)
        # ──────────────────────────────────────────────
        self.pkl_files: List[Path] = sorted(
            Path(pkl_dir).glob("*.pkl")
        )
        if not self.pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {pkl_dir}")

        # ──────────────────────────────────────────────
        # 3‑A) in‑memory 모드: feature 전체를 메모리에 적재
        # ──────────────────────────────────────────────
        if self.in_memory:
            feats, labels = [], []
            for fp in self.pkl_files:
                with fp.open("rb") as f:
                    feat_per_labels: Dict[str, Any] = pickle.load(f)
                # print(feat_per_labels)
                for text, mat in feat_per_labels.items():
                    if text not in self.text2idx:
                        # 알 수 없는 클래스는 건너뜀
                        continue
                    cls_idx = self.text2idx[text]

                    # mat: (N, D) numpy array | list
                    for vec in mat:
                        feats.append(torch.as_tensor(vec, dtype=dtype))
                        labels.append(cls_idx)

            self._features: List[torch.Tensor] = feats
            self._labels:   List[int]          = labels

        # ──────────────────────────────────────────────
        # 3‑B) lazy 모드: (file_idx, row_idx) 인덱스만 저장
        # ──────────────────────────────────────────────
        else:
            self._index: List[Tuple[int, str, int]] = []  # (file_i, text, row_i)
            self._cached_file: Tuple[int, Dict[str, Any]] | None = None

            for fi, fp in enumerate(self.pkl_files):
                with fp.open("rb") as f:
                    feat_per_labels: Dict[str, Any] = pickle.load(f)

                for text, mat in feat_per_labels.items():
                    if text not in self.text2idx:
                        continue
                    for row_i in range(len(mat)):
                        self._index.append((fi, text, row_i))

    # ──────────────────────────────────────────────
    # 필수 메서드
    # ──────────────────────────────────────────────
    def __len__(self) -> int:
        if self.in_memory:
            return len(self._features)
        else:
            return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.in_memory:
            return self._features[idx], self._labels[idx]

        # lazy 로딩: 필요한 벡터만 파일에서 꺼냄
        file_i, text, row_i = self._index[idx]
        cls_idx = self.text2idx[text]

        # 캐시된 파일 재사용
        if self._cached_file is None or self._cached_file[0] != file_i:
            with self.pkl_files[file_i].open("rb") as f:
                feat_per_labels = pickle.load(f)
            self._cached_file = (file_i, feat_per_labels)
        else:
            feat_per_labels = self._cached_file[1]

        vec = feat_per_labels[text][row_i]
        return torch.as_tensor(vec, dtype=self.dtype), cls_idx
