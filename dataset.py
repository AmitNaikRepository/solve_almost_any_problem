import torch 
import numpy as np 
from PIL import Image 
from PIL import ImageFile

#deal with the currput file as in image format 

Imagefile.LOAD_TRUNCATE_IMAGES = True

class ClassficationDataset:
    def __init__(self,image_path, targets,resize=None,augmentation=None):
        self.image_path=image_path
        self.targets=targets
        self.resize=resize
        self.augmentation=augmentation 


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,item):
        image=Image(self.image_path[item])
        #convert the image into rgb 
        image=image.conver('RGB')
        
        #correct target of the dataset 
        targets=self.targets[item]
        
        #lets deal with resize if we need resize image 

        if self.resize is not None:
            image =image.resize(self.resize[0],self.resize[1]),
            resample=Image.BILINEAR

        #now lets do the conversion with the convert 
        image =np.array(image)

        #so now lets handle the augemntation data 

        if self.augmentation is not None:
            augmented=self.augmentation(iamge=image)
            image=augmented['image']

        #now lets convert the CHW as pytorch accept where the HWS is store 

        image=np.transpose(image,(2,0,1)).astype(np.float32)


        #now lets return tensor of an image

        return {'image': torch.tensor(image,dtype=torch.float),
                'targets': torch.tensor(image,dtype=torch.long)}




