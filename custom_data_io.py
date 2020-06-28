from os.path import join as PJ
from torch.utils.data import Dataset
from PIL import Image
import torch
import csv
import pandas as pd

class myset(Dataset):
    def __init__(self, data_root, data_txt, class2idx, transform=None, delete_header = True):
        
        if data_txt[-3:] == 'csv':
            df = pd.read_csv(data_txt)
            if df.iloc[:,1].dropna().empty:
                df.iloc[:,1] = [key for key in class2idx.keys()][0]
                df.to_csv(data_txt, index=False)
            csv_file = data_txt
            txt_file = csv_file[:-3] + 'txt'
            with open(txt_file, "w") as my_output_file:
                with open(csv_file, "r") as my_input_file:
                    [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
                my_output_file.close()
            data_txt = txt_file
        
        self.data_root = data_root
        self.class2idx = class2idx
        self.transform = transform

        # Load data path
        with open(data_txt, 'r') as f:
            data = f.readlines()
            if delete_header == True:
                data = data[1:]
        self.data = [line.strip().split() for line in data]

    def __len__(self):
        """ Generate index list for __getitem__ """
        return len(self.data)

    def __getitem__(self, index):
        """ Call by DataLoader(an Iterator) """
        image_path, class_name = self.data[index]

        # Label (Type is torch.LongTensor for calculate loss)
        label = torch.LongTensor([self.class2idx[class_name]])

        # Load image and transform
        image = Image.open(PJ(self.data_root, image_path)).convert('RGB')
        image = self.transform(image) if self.transform else image
        return image, label
    def path_label(self, index):
        image_path, class_name = self.data[index]
        return image_path, class_name