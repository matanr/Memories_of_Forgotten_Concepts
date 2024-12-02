from torch.utils.data import Dataset
import json
import os
import os.path as osp
import pandas as pd


class CaptionsPaths:
    def __init__(self, root, *args, **kwargs):
        self.root = root
        images_in_nudity_dataset = os.listdir(osp.join(root, "imgs"))
        images_in_nudity_dataset.sort()
        all_images_indices = [
            int(p.replace("_0.png", "")) for p in images_in_nudity_dataset
        ]
        prompts_file_path = osp.join(root, "prompts.csv")
        assert osp.isfile(
            prompts_file_path
        ), f"Prompts file not found at {prompts_file_path}"
        prompts_df = pd.read_csv(prompts_file_path)

        def idx_to_prompt(image_idx):
            return prompts_df[prompts_df["case_number"] == image_idx].iloc[0].prompt

        self.images_paths = [
            osp.join(root, "imgs", x) for x in images_in_nudity_dataset
        ]
        self.prompts = [idx_to_prompt(idx) for idx in all_images_indices]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        return (self.images_paths[item], self.prompts[item])


class CocoCaptions17Paths(Dataset):
    """
    A PyTorch Dataset class for loading COCO 2017 captions dataset.

        root (str): Root directory of the COCO dataset.
        train (bool, optional): If True, loads the training dataset. If False, loads the validation dataset. Default is True.

    Attributes:
        train (bool): Indicates whether the dataset is for training or validation.
        images_root (str): Directory path to the images.
        captions_map (dict): A dictionary mapping image IDs to their corresponding captions.
        images_list (list): List of image filenames in the dataset.
        len (int): Number of images in the dataset.

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(item): Returns the image path and caption for the given index.
        _parse_captions(captions): Parses the captions from the annotations file.
        _get_single_item(item): Helper method to get a single item (image path and caption) by index.

    """

    def __init__(self, root, train=True):
        self.train = train
        if self.train:
            self.images_root = f"{root}/train2017"
            with open(f"{root}/annotations/captions_train2017.json") as f:
                self.captions_map = self._parse_captions(json.load(f)["annotations"])
        else:
            self.images_root = f"{root}/val2017"
            with open(f"{root}/annotations/captions_val2017.json") as f:
                self.captions_map = self._parse_captions(json.load(f)["annotations"])
        self.images_list = os.listdir(self.images_root)
        self.len = len(self.images_list)

    @staticmethod
    def _parse_captions(captions):
        captions_map = {}
        for d in captions:
            img_id = d["image_id"]
            if img_id not in captions_map:
                captions_map[img_id] = d["caption"]
        return captions_map

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self._get_single_item(item)

    def _get_single_item(self, item):
        image_path = self.get_image_path_by_index(item)
        image_id = int(self.images_list[item].split(".")[0])
        caption = self.captions_map[image_id]
        return image_path, caption

    def get_image_path_by_index(self, index):
        return os.path.join(self.images_root, self.images_list[index])
