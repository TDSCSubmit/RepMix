import os
from torchvision.transforms import transforms
import torchvision.datasets
import numpy as np
import copy
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
import robustness.datasets


class RepMNIST(torchvision.datasets.MNIST):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace mnist data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets

        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 28, 28)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 1))  # convert to HWC
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data

    
    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment mnist data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
      
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 28, 28)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 1))  # convert to HWC
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class RepKMNIST(torchvision.datasets.KMNIST):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace kmnist data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
      
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 28, 28)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 1))  # convert to HWC
       
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data


    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment kmnist data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 28, 28)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 1))  # convert to HWC
      
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets
 
        self.data = data


class RepCIFAR10(torchvision.datasets.CIFAR10):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace cifar10 data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        print('constraucting adv cifar10 testset')
        data = self.data
        # targets = self.targets

        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 32, 32)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC

        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data

    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment cifar10 data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 32, 32)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data             


class RepCIFAR100(torchvision.datasets.CIFAR100):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace cifar100 data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 32, 32)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
        
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data


    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment cifar100 data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
      
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 32, 32)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class RepImageNet(torchvision.datasets.ImageNet):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace imagenet data and targets with rep_xndarray,rep_yndarray ")

    # change this function
    def __getrepdataset__(self):

        data = self.data    #   ImageNet has no attribute named data
        # targets = self.targets
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 256, 256)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data

    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment imagenet data and targets with aug_xndarray,aug_yndarray ")
    
    # change this function
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 256, 256)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class RepLSUN(torchvision.datasets.LSUN):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace lsun data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 256, 256)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data

    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment lsun data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 256, 256)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
     
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class RepSVHN(torchvision.datasets.SVHN):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace svhn data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
      
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 32, 32)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray

        self.data = data

    def augdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment svhn data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 32, 32)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data    


class RepSTL10(torchvision.datasets.STL10):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace stl10 data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
      
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 96, 96)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray

        self.data = data

    def augdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment stl10 data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 96, 96)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data

#-------------------------------------------------------

class RepDataset:
    def __init__(self,args, custom_traindataset = None, custom_testdataset =None) -> None:
        print(f'initilize the dataset loading parameters')
        self._args = args 
        if custom_traindataset == None:
            # print("dataset from pytorch")
            self._traindataset = self.__loadtraindataset__()  
            self._testdataset = self.__loadtestdataset__() 
        else:
            # print("dataset from custom")
            self._traindataset = custom_traindataset
            self._testdataset = custom_testdataset

    def traindataset(self):
        return self._traindataset
    
    def testdataset(self):
        return self._testdataset

    def __loadtraindataset__(self):
        if self._args.dataset == 'mnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size            

            os.makedirs("/home/data/rep/mnist", exist_ok=True)
            train_dataset = RepMNIST(                                             
                "/home/data/rep/mnist",
                train=True,                                             
                download=True,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return train_dataset

        elif self._args.dataset == 'kmnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size            
            
            os.makedirs("/home/data/rep/kmnist", exist_ok=True)
            train_dataset = RepKMNIST(                                            
                "/home/data/rep/kmnist",
                train=True,                                             
                download=True,                                           
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            ) 
            return train_dataset

        elif self._args.dataset == 'cifar10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            print(f'load cifar10 dataset')
            os.makedirs("/home/data/rep/cifar10", exist_ok=True)
            train_dataset = RepCIFAR10(                                             
                "/home/data/rep/cifar10",
                train=True,                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        # transforms.Resize(self._args.img_size), 
                        # transforms.CenterCrop(self._args.img_size),
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]
                ),
            )
            # print('train_dataset:',train_dataset)
            # print('train_dataset.__dict__',train_dataset.__dict__)
            return train_dataset

        elif self._args.dataset == 'cifar100':
            os.makedirs("/home/data/rep/cifar100", exist_ok=True)  
            
            train_dataset = RepCIFAR100(                                            
                "/home/data/rep/cifar100",
                train=True,                                             
                download=True,                                           
                transform=transforms.Compose(
                    [
                        transforms.Resize(self._args.img_size), 
                        transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]
                ),
            )
            return train_dataset

        elif self._args.dataset == 'imagenet':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size # 256

            os.makedirs("/home/data/ImageNet", exist_ok=True)
            train_dataset = RepImageNet(                                             
                "/home/data/ImageNet",
                split='train',
                download=False,
                transform=transforms.Compose(                               
                    [
                        transforms.Resize(crop_size),                       
                        transforms.CenterCrop(crop_size),                    
                        transforms.ToTensor(),                                   
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]     
                ),
            ) 
            return train_dataset

        elif self._args.dataset == 'imagenetmixed10':
            in_path = "/home/data/ImageNet"              
            in_info_path = "/home/data/ImageNet/info"
            in_hier = ImageNetHierarchy(in_path, in_info_path)                  

            superclass_wnid = common_superclass_wnid('mixed_10')            
            class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

            # num_workers =4
            # batch_size =1
            custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges)
            print("custom_dataset.__dict__.keys()",custom_dataset.__dict__.keys())
            # custom_dataset.__dict__.keys() dict_keys(['ds_name', 'data_path', 'num_classes', 'mean', 'std', 'transform_train', 'transform_test', 'custom_class', 'label_mapping', 'custom_class_args'])
            # train_loader, test_loader = custom_dataset.make_loaders(workers=num_workers, batch_size=batch_size)
            train_dataset = custom_dataset

            return train_dataset

        elif self._args.dataset == 'lsun':
            os.makedirs("/home/data/rep/lsun/20210413", exist_ok=True)
            train_dataset =RepLSUN(                                             
                "/home/data/rep/lsun/20210413",
                 
                classes=['church_outdoor_train','classroom_train','tower_train'],
                transform=transforms.Compose(                               
                    [
                        transforms.Resize(self._args.img_size),                       
                        transforms.CenterCrop(self._args.img_size),                    
                        transforms.ToTensor(),                                  
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]     
                ),
                target_transform = None
            ) 
            return train_dataset
        
        elif self._args.dataset == 'svhn':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            os.makedirs("/home/data/rep/svhn", exist_ok=True)
            train_dataset = RepSVHN(                                             
                "/home/data/rep/svhn",
                split='train',                                              
                download=True,                                           
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),                        
                        # transforms.Resize(self._args.img_size), 
                        # transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return train_dataset            

        elif self._args.dataset == 'stl10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size    

            os.makedirs("/home/data/rep/stl10", exist_ok=True)
            train_dataset = RepSTL10(                                             
                "/home/data/rep/stl10",
                split='train',                                              
                download=True,                                           
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return train_dataset

    def __loadtestdataset__(self):
        if self._args.dataset == 'mnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size    

            os.makedirs("/home/data/rep/mnist", exist_ok=True)
            test_dataset = RepMNIST(                                             
                "/home/data/rep/mnist",
                train=False,                                              
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'kmnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size               
            os.makedirs("/home/data/rep/kmnist", exist_ok=True)
            test_dataset = RepKMNIST(                                          
                "/home/data/rep/kmnist",
                train=False,                                              
                download=False,                                           
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'cifar10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            os.makedirs("/home/data/rep/cifar10", exist_ok=True)
            test_dataset = RepCIFAR10(                                             
                "/home/data/rep/cifar10",
                train=False,                                              
                download=False,                                           
                transform=transforms.Compose(
                    [
                        # transforms.Resize(self._args.img_size), 
                        # transforms.CenterCrop(self._args.img_size),
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),                        
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'cifar100':
            os.makedirs("/home/data/rep/cifar100", exist_ok=True)
            test_dataset = RepCIFAR100(                                             
                "/home/data/rep/cifar100",
                train=False,                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(self._args.img_size), 
                        transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'imagenet':

            os.makedirs("/home/data/ImageNet", exist_ok=True)

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size # 256
                # crop_size = 1024

            test_dataset = RepImageNet(                                             
                "/home/data/ImageNet",
                split='val',
                download=False,
                transform=transforms.Compose(                                
                    [
                        transforms.Resize(crop_size),                        
                        transforms.CenterCrop(crop_size),                    
                        transforms.ToTensor(),                                                     
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]     
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'imagenetmixed10':
            in_path = "/home/data/ImageNet"             
            in_info_path = "/home/data/ImageNet/info"
            in_hier = ImageNetHierarchy(in_path, in_info_path)                  

            superclass_wnid = common_superclass_wnid('mixed_10')            
            class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

            

            custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges)
           
            test_dataset = custom_dataset
 

            return test_dataset

        elif self._args.dataset == 'lsun':
            os.makedirs("/home/data/rep/lsun/20210413", exist_ok=True)
            test_dataset = RepLSUN(                                             
                "/home/data/rep/lsun/20210413",
                
                classes=['church_outdoor_test','classroom_test','tower_test'],
                transform=transforms.Compose(                               
                    [
                        transforms.Resize(self._args.img_size),                       
                        transforms.CenterCrop(self._args.img_size),                    
                        transforms.ToTensor(),                                  
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]     
                ),
                target_transform = None
            )
            return test_dataset    

        elif self._args.dataset == 'stl10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size                
            os.makedirs("/home/data/rep/stl10", exist_ok=True)
            test_dataset = RepSTL10(                                             
                "/home/data/rep/stl10",
                split='test',                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )  
            return test_dataset    
        
        elif self._args.dataset == 'svhn':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            os.makedirs("/home/data/rep/svhn", exist_ok=True)
            test_dataset = RepSVHN(                                             
                "/home/data/rep/svhn",
                split='test',                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),                        
                        # transforms.Resize(self._args.img_size), 
                        # transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )  
            return test_dataset  

class Array2Dataset:
    def __init__(self, args, x_ndarray, y_ndarray, ori_dataset: RepCIFAR10):
         
        self._args = args
        self._x_ndarray = x_ndarray
        self._y_ndarray = y_ndarray
        self._ori_dataset_4_rep = copy.deepcopy(ori_dataset)
        self._ori_dataset_4_aug = copy.deepcopy(ori_dataset)

    def repdataset(self)->"RepDataset(torchvision.datasets.__dict__)":   
        self._rep_dataset = self.__getrepdataset__()
        return self._rep_dataset

    def augdataset(self) ->"RepDataset(torchvision.datasets.__dict__)":
        self._aug_dataset = self.__getaugdataset__()
        return self._aug_dataset
    
    def __getrepdataset__(self):

        # print("before rep, self._ori_dataset_4_rep.data:",self._ori_dataset_4_rep.data[:2])
        self._ori_dataset_4_rep.replacedataset(self._x_ndarray,self._y_ndarray)
        # print("after rep, self._ori_dataset_4_rep.data:",self._ori_dataset_4_rep.data[:2])
        # return rep_dataset
        return self._ori_dataset_4_rep
    
    def __getaugdataset__(self)->"RepDataset(torchvision.datasets.__dict__)":
        self._ori_dataset_4_aug.augmentdataset(self._x_ndarray,self._y_ndarray)
        # return aug_dataset
        return self._ori_dataset_4_aug

