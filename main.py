if __name__ == '__main__':
    from models.network import *
    import argparse
    import os
    from os.path import join as PJ
    from RandAugment import RandAugment
    from custom_data_io import myset
    import torch.optim as optim
    from DML import DML

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, default='exp_1')
    args = parser.parse_args()

    config_path = PJ(os.getcwd(), f"{args.exp}.yaml")
    config = config(config_path)
    exp_name = config['exp_name']
    print(f"EXP: {exp_name}")

    train_transforms = transforms.Compose([transforms.Resize((385, 505)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize((385, 505)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    train_transforms.transforms.insert(0, RandAugment(2, 14))

    class2idx = config['class2idx']
    batch = config['batch_size']
    num_workers = config['num_workers']

    train_data = datasets.ImageFolder(root = config['Train'], transform = train_transforms)
    valid_data = myset(config['Val'], config['val_txt'], class2idx, test_transforms)
    test_data = myset(config['Test'], config['test_txt'], class2idx, test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch , num_workers = num_workers,shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch,  num_workers = num_workers,shuffle = False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch,  num_workers = num_workers,shuffle = False)

    ms = []
    model1 = res(3)
    ms.append(model1)
    model2 = mob(3)
    ms.append(model2)
    model3 = alex(3)
    ms.append(model3)

    optimizer = optim.Adam([{'params': model1.parameters()}, {'params': model2.parameters()}, {'params': model3.parameters()}], lr = 0.00001, weight_decay=1e-3)

    dml = DML(ms, optimizer, parallel = True)

    dml.train(300, train_loader, valid_loader)