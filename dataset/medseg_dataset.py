import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def organize_data():
    root = './MedSeg/covid-segmentation'
    X1 = np.load(os.path.join(root, 'images_medseg.npy'))
    Y1 = np.load(os.path.join(root, 'masks_medseg.npy'))
    X2 = np.load(os.path.join(root, 'images_radiopedia.npy'))
    Y2 = np.load(os.path.join(root, 'masks_radiopedia.npy'))
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:int(0.8*len(X))], indices[int(0.8*len(X)):]

    with open(os.path.join(root, 'train_images.npy'), 'wb') as f:
        np.save(f, X[train_idx])

    with open(os.path.join(root, 'train_masks.npy'), 'wb') as f:
        np.save(f, Y[train_idx])

    with open(os.path.join(root, 'val_images.npy'), 'wb') as f:
        np.save(f, X[val_idx])

    with open(os.path.join(root, 'val_masks.npy'), 'wb') as f:
        np.save(f, Y[val_idx])


class seg_dataset(Dataset):
    def __init__(self, split, transform=None):
        self.root = './MedSeg/covid-segmentation'

        if split == 'train':
            self.X = np.load(os.path.join(self.root, 'train_images.npy'))
            self.Y = np.load(os.path.join(self.root, 'train_masks.npy'))
        elif split == 'val':
            self.X = np.load(os.path.join(self.root, 'val_images.npy'))
            self.Y = np.load(os.path.join(self.root, 'val_masks.npy'))
        else:
            self.X = np.load(os.path.join(self.root, 'test_images_medseg.npy'))
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x *= 255.0/x.max()
        x = np.stack((x,)*3, axis=3)
        x = x.astype(np.uint8)
        x = x.squeeze()
        y0 = y[0, :, :]
        y1 = y[1, :, :]
        if self.transform is not None:
            im_x = Image.fromarray(x)
            im_y0 = Image.fromarray(y0)
            im_y1 = Image.fromarray(y1)
            x = self.transform(im_x)
            y0 = self.transform(im_y0)
            y1 = self.transform(im_y1)

        return x, y0, y1, index

    def __len__(self):
        return len(self.X)


# train_transform = transforms.Compose([])
# train_transform.transforms.append(transforms.RandomHorizontalFlip(1))
# train_transform.transforms.append(transforms.ToTensor())
# train_dataset = seg_dataset('train', train_transform)
#
# trainLoader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#
# for image, mask0, mask1, idx in enumerate(trainLoader):
#     image = image.numpy()
#     plt.imshow(image)
#     break

