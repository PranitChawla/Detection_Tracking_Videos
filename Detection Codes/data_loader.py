import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image
import json


class Dataset(data.Dataset):

    def __init__(self, root, data_file_name, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data: index file name.
            transform: image transformer.
        """
        self.root = root
        with open(data_file_name, 'r') as f:
            self.data = json.load(f)
        self.ids = os.listdir(root)
#         print(len(self.data), data_file_name)
        self.transform = transform
        
    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        id = self.ids[index]
        file_name = id.split("_")[0]
        intervals = data[file_name]
        # print (id)
        candidate_video = []
        # print (id)
        labels = 0 
        frame_number = id.split("_")[-1]
        frame_number = frame_number.split('.')[0]
        # print (frame_number)
        test = []
        for i in range (int(frame_number),int(int(frame_number)+6)):
            # print (i)
            candidate_img_name = file_name+'_'+str(i)
            candidate_img_name += '.png'
            # print (candidate_img_name)
            test.append(candidate_img_name)
            try:
                candidate_image = Image.open(os.path.join(self.root, candidate_img_name)).convert('RGB')
                if self.transform is not None:
                    candidate_image = self.transform(candidate_image)
                print (candidate_image.shape)
                # print (type(candidate_image))
            except:
                print ("erro")
                candidate_image = torch.zeros(120,150)
                # candidate_image = 
            # candidate_image = candidate_image[:,360:480,250:400]
            for interval in intervals:
                if i>=interval[0] and i<=interval[1]:
                    labels+=1
                    break
            # print (type(candidate_image))
            candidate_video.append(candidate_image)
            

        candidate_video = torch.stack(candidate_video)

        candidate_video = candidate_video.permute(1,0,2,3)

        # print (candidate_video.shape)

        if (labels>=3):
            candidate_label = 1
        else:
            candidate_label = 0

        


        # print (candidate_image.shape)
        # print (candidate_video.shape,candidate_label)
        


        return candidate_label, candidate_video

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of images.
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    candidate_label, candidate_video = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    candidate_video = torch.stack(candidate_video,0)

    # candidate_images = torch.stack(candidate_images, 0)

    candidate_labels = [label for label in candidate_label]

    candidate_labels = torch.FloatTensor(candidate_labels)



    # Merge captions (from tuple of 1D tensor to 2D tensor).
    return candidate_labels,candidate_video


def get_loader(root, data_file_name, transform, batch_size, shuffle, return_target, num_workers):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    # relative caption dataset
    dataset = Dataset(root=root,
                      data_file_name=data_file_name,
                      transform=transform)
#     print(len(dataset))    
    # Data loader for the dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              timeout=60)

    return data_loader


