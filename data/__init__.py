from importlib import import_module
from torchvision import transforms
from utils.random_erasing import RandomErasing
from data.sampler import RandomSampler
from torch.utils.data import dataloader

class Data:
    def __init__(self, args):

        train_list = [
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if args.random_erasing:
            train_list.append(RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, 'train')
            self.train_loader = dataloader.DataLoader(self.trainset,
                            sampler=RandomSampler(self.trainset,args.batchid,batch_image=args.batchimage),
                            #shuffle=True,
                            batch_size=args.batchid * args.batchimage,
                            num_workers=args.nThread)
        else:
            self.train_loader = None
        
        if args.data_test in ['Market1501']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')

        else:
            raise Exception()

        self.test_loader = dataloader.DataLoader(self.testset, batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=args.batchtest, num_workers=args.nThread)
        