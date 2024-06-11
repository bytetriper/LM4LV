#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import json
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Union,Iterable

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

class LM4LVDataset(Dataset):
    """LM4LV Dataset"""

    def __init__(self, data_file_path: str, vision_root_path: Union[str,list[str]], vision_type="image"):
        """Initialize supervised datasets

        :param str data_file_path: path of conversation file path
        :param str vision_root_path: vision root path
        :param str vision_type: type of vision data, defaults to 'image', image / pcl
        """
        super(LM4LVDataset, self).__init__()
        self.vision_type = vision_type
        with open(data_file_path, "r") as fr:
            json_data = json.load(fr)
        if not isinstance(vision_root_path, Iterable):
            vision_root_path = [vision_root_path]
        self.vision_files = []
        self.info_dict = {}
        for root_path in vision_root_path:
            potential_image_npy_path = os.path.join(root_path, "images.npy")
            if os.path.exists(potential_image_npy_path):
                print(f"🏃‍♀️ Use preloading strategy for vision data, from {potential_image_npy_path}")
                info_dict_path = os.path.join(root_path, "info_dict.npy")
                vision_files = np.load(potential_image_npy_path) # vision_file: list of npy files
                info_dict = np.load(info_dict_path, allow_pickle=True).item()# info_dict: dict of {image_path: index}
                previous_len = len(self.vision_files)
                # merge 
                self.vision_files.extend(vision_files)
                #update idx in info_dict by adding the length of the previous files
                for key in info_dict:
                    info_dict[key] += previous_len
                self.info_dict.update(info_dict) # if data name conflict, the later one will overwrite the previous one
                #print(self.info_dict)
            else:
                print(f"💤 Use lazy loading strategy for vision data, from {root_path}")
        self.vision_path_list, self.caption_list, self.task_type_list = [], [], []
        for item in json_data:
            if not vision_type in item:
                continue
            one_vision_name, one_caption = item[vision_type], item["conversations"]
            task_type = item["task_type"] if "task_type" in item else "normal"
            one_vision_path = []
            if isinstance(one_vision_name, str):
                one_vision_name = [one_vision_name]
            for vision_instance in one_vision_name:
                if  vision_type == 'image' and  (not vision_instance.startswith("/")):
                    one_vision_path.append(os.path.join(vision_root_path, vision_instance))
                else:
                    one_vision_path.append(vision_instance)

            self.vision_path_list.append(one_vision_path)
            self.caption_list.append(one_caption)
            self.task_type_list.append(task_type)
        print(f"📜 collect {len(self.vision_path_list)} samples")

    def __len__(self):
        """get dataset length

        :return int: length of dataset
        """
        return len(self.vision_path_list)

    def __getitem__(self, i):
        """get one sample"""
        vision_file = []
        for path in self.vision_path_list[i]:
            try_get_file = self.info_dict.get(path, None)
            vision_file.append(try_get_file)
        return dict(
            vision_path=self.vision_path_list[i],
            output_text=self.caption_list[i],
            vision_type=self.vision_type,
            task_type=self.task_type_list[i],
            vision_file= vision_file,
        )

    def collate(self, instances):
        """collate function for dataloader"""
        vision_paths, output_texts, task_type, vision_files = tuple(
            [instance[key] for instance in instances]
            for key in ("vision_path", "output_text", "task_type","vision_file")
        )
        return dict(
            vision_paths=vision_paths,
            output_texts=output_texts,
            vision_type=self.vision_type,
            task_type=task_type,
            vision_files=vision_files,
        )
