import json
import os
import numpy as np
import torch
import random
from torchvision.datasets.folder import default_loader
from data.datasets.InternalData import replace_img_ext
from data.datasets.InternalData_ms import InternalDataMSSigma, get_closest_ratio
from data.builder import get_data_path, DATASETS
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from data.datasets.utils import *

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

@DATASETS.register_module()
class FlexibleInternalDataMS(InternalDataMSSigma):
    def __init__(self,
                 roots,       # a list of root that has the same length as image_list_json_lst
                 json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
                 transform=None,
                 resolution=256,
                 sample_subset=None,
                 load_vae_feat=False,
                 load_t5_feat=False,
                 input_size=32,
                 patch_size=2,
                 mask_ratio=0.0,
                 mask_type='null',
                 load_mask_index=False,
                 real_prompt_ratio=0.0,
                 max_length=300,
                 return_image_id=False,
                 config=None,
                 **kwargs):

        roots = [get_data_path(root) for root in roots]
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.load_t5_feat = load_t5_feat
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.N = int(resolution // (input_size // patch_size))
        self.mask_ratio = mask_ratio
        self.load_mask_index = load_mask_index
        self.mask_type = mask_type
        self.real_prompt_ratio = real_prompt_ratio
        self.max_length = max_length
        self.base_size = int(kwargs['aspect_ratio_type'].split('_')[-1])
        self.aspect_ratio = eval(kwargs.pop('aspect_ratio_type'))       # base aspect ratio
        self.return_image_id = return_image_id

        self.meta_data_clean = []
        self.img_samples = []
        self.txt_samples = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        self.mask_index_samples = []
        self.ratio_index = {}
        self.ratio_nums = {}

        self.weight_dtype = torch.float16 if self.real_prompt_ratio > 0 else torch.float32
        self.interpolate_model = InterpolationMode.BICUBIC
        if self.aspect_ratio in [ASPECT_RATIO_2048, ASPECT_RATIO_2880]:
            self.interpolate_model = InterpolationMode.LANCZOS
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []     # used for self.getitem
            self.ratio_nums[float(k)] = 0      # used for batch-sampler

        if not json_lst:
            json_lst = [os.path.join(root, 'meta_data.json') for root in roots]

        for root, json_file in zip(roots, json_lst):

            meta_data = self.load_json(os.path.join(root, json_file))
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4.5]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([
                os.path.join(root, item['image_path']) for item in meta_data_clean
            ])
            # self.txt_samples.extend([item['prompt'] for item in meta_data_clean])
            # self.sharegpt4v_txt_samples.extend([item['sharegpt4v'] if 'sharegpt4v' in item else '' for item in meta_data_clean])
            self.txt_samples.extend([item['caption'] for item in meta_data_clean])

        # Set loader and extensions
        if load_vae_feat:
            self.transform = None
            self.loader = self.vae_feat_loader
        else:
            self.loader = default_loader

        if sample_subset is not None:
            self.sample_subset(sample_subset)  # sample dataset for local debug

        # scan the dataset for ratio static
        for i, info in enumerate(self.meta_data_clean[:len(self.meta_data_clean)//3]):
            ori_h, ori_w = info['height'], info['width']
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
            self.ratio_nums[closest_ratio] += 1
            if len(self.ratio_index[closest_ratio]) == 0:
                self.ratio_index[closest_ratio].append(i)

    def getdata(self, index):
        img_path = self.img_samples[index]
        # real_prompt = random.random() < self.real_prompt_ratio
        # npz_path = self.txt_feat_samples[index] if real_prompt else self.gpt4v_txt_feat_samples[index]
        txt = self.txt_samples[index]
        # npy_path = self.vae_feat_samples[index]
        data_info = {}
        ori_h, ori_w = self.meta_data_clean[index]['height'], self.meta_data_clean[index]['width']

        # Calculate the closest aspect ratio and resize & crop image[w, h]
        closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
        closest_size = list(map(lambda x: int(x), closest_size))
        self.closest_ratio = closest_ratio

        if self.load_vae_feat:
            # img = self.loader(npy_path)
            # if index not in self.ratio_index[closest_ratio]:
            #     self.ratio_index[closest_ratio].append(index)
            # h, w = (img.shape[1], img.shape[2])
            # assert h, w == (ori_h//8, ori_w//8)
            pass # not used for now
        else:
            img = self.loader(img_path)
            h, w = (img.size[1], img.size[0])
            assert h, w == (ori_h, ori_w)

        data_info['img_hw'] = torch.tensor([ori_h, ori_w], dtype=torch.float32)
        data_info['aspect_ratio'] = closest_ratio
        data_info["mask_type"] = self.mask_type

        if self.return_image_id:
            assert 'image_id' in self.meta_data_clean[index]
            data_info['image_id'] = self.meta_data_clean[index]['image_id']

        attention_mask = torch.ones(1, 1, self.max_length)
        if self.load_t5_feat:
            # txt_info = np.load(npz_path)
            # txt_fea = torch.from_numpy(txt_info['caption_feature'])
            # if 'attention_mask' in txt_info.keys():
            #     attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
            # if txt_fea.shape[1] != self.max_length:
            #     txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_length-txt_fea.shape[1], 1)], dim=1).to(self.weight_dtype)
            #     attention_mask = torch.cat([attention_mask, torch.zeros(1, 1, self.max_length-attention_mask.shape[-1])], dim=-1)
            pass # not used for now
        else:
            txt_fea = txt

        if not self.load_vae_feat:
            if closest_size[0] / ori_h > closest_size[1] / ori_w:
                resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
            else:
                resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(resize_size, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(closest_size),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        if self.transform:
            img = self.transform(img)

        return img, txt_fea, attention_mask.to(torch.int16), data_info

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.ratio_index[self.closest_ratio])
        raise RuntimeError('Too many bad data.')




@DATASETS.register_module()
class FlexibleInternalDataMSML(InternalDataMSSigma):
    def __init__(self,
                 roots,       # a list of root that has the same length as image_list_json_lst
                 json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
                 transform=None,
                 resolution=256,
                 sample_subset=None,
                 load_vae_feat=False,
                 load_t5_feat=False,
                 input_size=32,
                 patch_size=2,
                 mask_ratio=0.0,
                 mask_type='null',
                 load_mask_index=False,
                 real_prompt_ratio=0.0,
                 max_length=300,
                 config=None,
                 **kwargs):

        roots = [get_data_path(root) for root in roots]
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.load_t5_feat = load_t5_feat
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.N = int(resolution // (input_size // patch_size))
        self.mask_ratio = mask_ratio
        self.load_mask_index = load_mask_index
        self.mask_type = mask_type
        self.real_prompt_ratio = real_prompt_ratio
        self.max_length = max_length
        self.base_size = int(kwargs['aspect_ratio_type'].split('_')[-1])
        self.aspect_ratio = eval(kwargs.pop('aspect_ratio_type'))       # base aspect ratio

        self.meta_data_clean = []
        self.img_samples = []
        self.txt_samples = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        self.mask_index_samples = []
        self.ratio_index = {}
        self.ratio_nums = {}

        self.weight_dtype = torch.float16 if self.real_prompt_ratio > 0 else torch.float32
        self.interpolate_model = InterpolationMode.BICUBIC
        if self.aspect_ratio in [ASPECT_RATIO_2048, ASPECT_RATIO_2880]:
            self.interpolate_model = InterpolationMode.LANCZOS
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []     # used for self.getitem
            self.ratio_nums[float(k)] = 0      # used for batch-sampler

        if not json_lst:
            json_lst = [os.path.join(root, 'meta_data.json') for root in roots]

        for root, json_file in zip(roots, json_lst):

            meta_data = self.load_json(os.path.join(root, json_file))
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4.5]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([
                os.path.join(root, item['image_path']) for item in meta_data_clean
            ])
            # self.txt_samples.extend([item['prompt'] for item in meta_data_clean])
            # self.sharegpt4v_txt_samples.extend([item['sharegpt4v'] if 'sharegpt4v' in item else '' for item in meta_data_clean])
            self.txt_samples.extend([item['caption'] for item in meta_data_clean])

        # Set loader and extensions
        if load_vae_feat:
            self.transform = None
            self.loader = self.vae_feat_loader
        else:
            self.loader = default_loader

        if sample_subset is not None:
            self.sample_subset(sample_subset)  # sample dataset for local debug

        # scan the dataset for ratio statistics
        for i, info in enumerate(self.meta_data_clean[:len(self.meta_data_clean)//3]):
            ori_h, ori_w = info['height'], info['width']
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
            self.ratio_nums[closest_ratio] += 1
            if len(self.ratio_index[closest_ratio]) == 0:
                self.ratio_index[closest_ratio].append(i)

    def getdata(self, index):
        img_path = self.img_samples[index]
        # real_prompt = random.random() < self.real_prompt_ratio
        # npz_path = self.txt_feat_samples[index] if real_prompt else self.gpt4v_txt_feat_samples[index]
        txt = self.txt_samples[index]
        # npy_path = self.vae_feat_samples[index]
        data_info = {}
        ori_h, ori_w = self.meta_data_clean[index]['height'], self.meta_data_clean[index]['width']

        # Calculate the closest aspect ratio and resize & crop image[w, h]
        closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
        closest_size = list(map(lambda x: int(x), closest_size))
        self.closest_ratio = closest_ratio

        if self.load_vae_feat:
            # img = self.loader(npy_path)
            # if index not in self.ratio_index[closest_ratio]:
            #     self.ratio_index[closest_ratio].append(index)
            # h, w = (img.shape[1], img.shape[2])
            # assert h, w == (ori_h//8, ori_w//8)
            pass # not used for now
        else:
            img = self.loader(img_path)
            h, w = (img.size[1], img.size[0])
            assert h, w == (ori_h, ori_w)

        data_info['img_hw'] = torch.tensor([ori_h, ori_w], dtype=torch.float32)
        data_info['aspect_ratio'] = closest_ratio
        data_info["mask_type"] = self.mask_type

        attention_mask = torch.ones(1, 1, self.max_length)
        if self.load_t5_feat:
            # txt_info = np.load(npz_path)
            # txt_fea = torch.from_numpy(txt_info['caption_feature'])
            # if 'attention_mask' in txt_info.keys():
            #     attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
            # if txt_fea.shape[1] != self.max_length:
            #     txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_length-txt_fea.shape[1], 1)], dim=1).to(self.weight_dtype)
            #     attention_mask = torch.cat([attention_mask, torch.zeros(1, 1, self.max_length-attention_mask.shape[-1])], dim=-1)
            pass # not used for now
        else:
            txt_fea = txt

        if not self.load_vae_feat:
            if closest_size[0] / ori_h > closest_size[1] / ori_w:
                resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
            else:
                resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(resize_size, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(closest_size),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        if self.transform:
            img = self.transform(img)

        return img, txt_fea, attention_mask.to(torch.int16), data_info

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.ratio_index[self.closest_ratio])
        raise RuntimeError('Too many bad data.')
    

    def tokenize_caption(self, idx):
        txt = self.txt_samples[index]

        return txt
