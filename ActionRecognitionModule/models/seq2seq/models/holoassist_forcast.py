import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader.tsv_multimodal import MultiTSVDataset
from typing import List
from src.data_utils import BlobStorage, disk_usage, generate_lineidx
import random
import numpy as np
from typing import (
    Callable,
    List,
    Tuple,
    Union,
    NamedTuple,
    Optional,
    Dict,
)


class HoloAssistForacast(MultiTSVDataset):
    # Dataset class that take 30sec(30 *30) input return 5sec output (30*5)

    def __init__(
        self,
        root: str,
        tasks: List[str],
        time_range: int = 0,
        time_out_range: int = 0,
        max_samples: int = 1,
        max_out_samples: int = 1,
        subset: str = None,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Callable = None,
        target_transform: Optional[Callable] = None,
        label_file: str = None,
        token_file: str = None,
        is_train: bool = True,
        azcopy_path: str = None,
        prefixes: Optional[Dict[str, str]] = None,
        max_images: Optional[int] = None,
        version: str = "ember_v0.5.1,0917",
        reference: str = "rgb",
        hand_norm: bool = False,
        eye_norm: bool = False,
    ):
        super().__init__(
            root,
            tasks,
            time_range,
            max_samples,
            subset,
        )

        ### Make the list of data for RNN
        self.max_out_samples = max_out_samples

    def __len__(self):
        return len(self._input_list)

    def __getitem__(self, idx):
        # TODO: Provide clean hand pose, normalized to the head pose.
        ret_i = []
        ret_o = []
        pos = self._input_list[idx]
        pos_o = self._output_list[idx]

        hand_xyz = []
        for ii in range(26):
            hand_xyz += [3 + ii * 16, 7 + ii * 16, 11 + ii * 16]

        for p in pos:
            self._fp.seek(p)
            ret_i_temp = [s.strip() for s in self._fp.readline().split("\t")][
                3 : 3 + 16 * 26
            ]

            ret_i.append(np.array(ret_i_temp).astype(np.float32)[hand_xyz])
        for p_o in pos_o:
            self._fp.seek(p_o)
            ret_o_temp = [s.strip() for s in self._fp.readline().split("\t")][
                3 : 3 + 16 * 26
            ]
            ret_o.append(np.array(ret_o_temp).astype(np.float32)[hand_xyz])

        ret_o = np.array(ret_o)
        ret_i = np.array(ret_i)
        return torch.tensor(ret_i).to(self.device), torch.tensor(ret_o).to(
            self.device
        )
