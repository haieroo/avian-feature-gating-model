import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_by_stem(root, dir_name, image_name, prefer_ext=None):
    """
    根据 texture 的 image_name，在 shape/color 文件夹中寻找同 stem 文件。
    例如:
        image_name = xxx.jpg
        shape 可以自动找到 xxx.png
        color 可以自动找到 xxx.jpg
    """
    stem = os.path.splitext(image_name)[0]

    if prefer_ext is not None:
        cand = os.path.join(root, dir_name, stem + prefer_ext)
        if os.path.exists(cand):
            return cand

    # 先尝试原始文件名
    cand = os.path.join(root, dir_name, image_name)
    if os.path.exists(cand):
        return cand

    # 再尝试常见后缀
    for ext in IMG_EXTS:
        cand = os.path.join(root, dir_name, stem + ext)
        if os.path.exists(cand):
            return cand

    return None


class My_Dataset(Dataset):
    def __init__(self, root_shape, root_texture, root_color):
        super().__init__()
        self.root_shape = root_shape
        self.root_texture = root_texture
        self.root_color = root_color

        dirs = os.listdir(self.root_texture)
        self.label_list = sorted(dirs)

        self.image_name = []
        self.dir_name = []
        self.image_label = {}

        label = -1
        idx = -1

        missing_shape = 0
        missing_color = 0

        for dir_name in sorted(dirs):
            label += 1
            texture_dir = os.path.join(self.root_texture, dir_name)
            files = os.listdir(texture_dir)

            for file in sorted(files):
                if os.path.splitext(file)[1].lower() not in IMG_EXTS:
                    continue

                shape_path = find_by_stem(
                    self.root_shape,
                    dir_name,
                    file,
                    prefer_ext=".png"
                )
                color_path = find_by_stem(
                    self.root_color,
                    dir_name,
                    file,
                    prefer_ext=".jpg"
                )

                if shape_path is None:
                    missing_shape += 1
                    continue

                if color_path is None:
                    missing_color += 1
                    continue

                idx += 1
                self.image_name.append(file)
                self.dir_name.append(dir_name)
                self.image_label[idx] = label

        print(f"[DATA] root_texture = {self.root_texture}")
        print(f"[DATA] classes      = {len(self.label_list)}")
        print(f"[DATA] valid images = {len(self.image_name)}")
        print(f"[DATA] missing shape = {missing_shape}")
        print(f"[DATA] missing color = {missing_color}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_images(self, index):
        image_name = self.image_name[index]
        img_dir = self.dir_name[index]
        image_label = int(self.image_label[index])

        texture_img_path = os.path.join(self.root_texture, img_dir, image_name)

        shape_img_path = find_by_stem(
            self.root_shape,
            img_dir,
            image_name,
            prefer_ext=".png"
        )

        color_img_path = find_by_stem(
            self.root_color,
            img_dir,
            image_name,
            prefer_ext=".jpg"
        )

        if shape_img_path is None:
            raise FileNotFoundError(f"Missing shape image for {img_dir}/{image_name}")

        if color_img_path is None:
            raise FileNotFoundError(f"Missing color image for {img_dir}/{image_name}")

        texture_img = Image.open(texture_img_path).convert("RGB")
        shape_img = Image.open(shape_img_path).convert("RGB")
        color_img = Image.open(color_img_path).convert("RGB")

        texture_img = self.transform(texture_img)
        shape_img = self.transform(shape_img)
        color_img = self.transform(color_img)

        return texture_img, shape_img, color_img, image_label, image_name

    def __getitem__(self, index):
        return self.load_images(index)

    def __len__(self):
        return len(self.image_name)


def get_Dataloader(root_shape, root_texture, root_color, batch_size, shuffle=True):
    return DataLoader(
        My_Dataset(root_shape, root_texture, root_color),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )