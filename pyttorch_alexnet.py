import torch
import torch.nn as nn
import torch.nn.functional as F 


class AlexNet:
    def __init__(self):
        super(AlexNet,self).__init__()
        #lets make the convolution part of nn 

        self.conv1=nn.Conv2d(
            in_channels=3,out_channels=96,kernel_size=5,stride=1,padding=2
        )
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2=nn.Conv2d(in_channels=3,out_channels=96,kernel_size=5,stride=1,padding=2)
        self.pool2=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3=nn.Conv2d(in_channels=3,out_channels=96,kernel_size=3,stride=2)
        self.conv4=nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)


        self.conv5=nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)


        self.pool3=nn.MaxPool2d(kernel_size=3,stride=2)
        #dense layer part 

        self.fc1=nn.Linear(in_feature=9216,out_features=4049)

        self.dropout1=nn.Dropout(0.5)
        self.fc2=nn.Linear(in_feature=4049,out_features=1000)

    def forward(self,image):
        #get the width , batch ,channel , height 
        #input batch image 
        #original size of the image(bs,3,227,227)

        bs,c,h,w=image.size()
        x=F.relu(self.conv1(image))#size (bs,95,55,55)
        x=self.pool1(x)
        x=f.relu(self.conv2(x))
        x=self.pool2(x)
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.relu(self.conv5(x))
        x=self.pool3(x)
        x=x.view(bs,-1)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x = F.relu(self.fc2(x)) # size: (bs, 4096)
        x = self.dropout2(x) # size: (bs, 4096)
        x = F.relu(self.fc3(x))
        x=torch.softmax(x,axis=1)
        return x 


