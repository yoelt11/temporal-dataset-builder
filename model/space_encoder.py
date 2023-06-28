import torch
from torch import nn

class SpaceEncoder(nn.Module):

    def __init__(self, channel_in, latent_dim):
        super().__init__()
        
        #-- normalize data
        #self.batch_norm = nn.BatchNorm2d(channel_in)

        # middle_layer >  latent dimension
        middle_layers =  16
        #-- feature extraction
        self.conv_1 = nn.Conv2d(channel_in, middle_layers, kernel_size=5, stride=2, padding=2)
        self.act_1 = nn.LeakyReLU(inplace=True)
        #--
        self.conv_2 = nn.Conv2d(middle_layers, middle_layers, kernel_size=5, stride=2, padding=2)
        self.act_2 = nn.LeakyReLU(inplace=True)
        #--
        self.conv_3 = nn.Conv2d(middle_layers, middle_layers, kernel_size=5, stride=2, padding=2)
        self.act_3 = nn.LeakyReLU(inplace=True)
        #--
        self.conv_4 = nn.Conv2d(middle_layers, middle_layers, kernel_size=5, stride=2, padding=2)
        self.act_4 = nn.LeakyReLU(inplace=True)
        #--
        self.conv_5 = nn.Conv2d(middle_layers, middle_layers, kernel_size=5, stride=2, padding=2)
        self.act_5 = nn.LeakyReLU(inplace=True)
        
        #-- scale combination
        self.reduction_1 = nn.Linear(32*128,middle_layers//2)
        self.act_6 = nn.LeakyReLU(inplace=True)

        self.reduction_2 = nn.Linear(16*64, middle_layers//2)
        self.act_7 = nn.LeakyReLU(inplace=True)

        self.reduction_3 = nn.Linear(8*32, middle_layers//2)
        self.act_8 = nn.LeakyReLU(inplace=True)

        self.reduction_4 = nn.Linear(4*16, middle_layers//2)
        self.act_9 = nn.LeakyReLU(inplace=True)

        self.reduction_5 = nn.Linear(2*8, middle_layers//2)
        self.act_10 = nn.LeakyReLU(inplace=True)

        # reduction to latent_dim
        self.linear = nn.Linear(latent_dim*middle_layers//2, latent_dim)
        self.softmax = nn.Sigmoid()


    def forward(self, X):
        # batch normalization
        #X = self.batch_norm(X.permute(0,3,1,2)) # B=300, H=64, W= 256, C= 3
        X = X.permute(0,3,1,2) # B=300, H=64, W= 256, C= 3
        # extract features at different scales
        scale_1 = self.act_1(self.conv_1(X)) # B=300, C=6, H=30, W= 126
        scale_2 = self.act_2(self.conv_2(scale_1)) # B=300, C=12, H=13, W= 61
        scale_3 = self.act_3(self.conv_3(scale_2)) # B=300, C=8, H=8, W= 32
        scale_4 = self.act_4(self.conv_4(scale_3)) # B=300, C=8, H=8, W= 32
        scale_5 = self.act_5(self.conv_5(scale_4)) # B=300, C=16, H=2, W= 8
        # scale combination
        scale_combination = (self.act_1(self.reduction_1(scale_1.flatten(2))) + 
                             self.act_2(self.reduction_2(scale_2.flatten(2))) + 
                             self.act_3(self.reduction_3(scale_3.flatten(2))) + 
                             self.act_4(self.reduction_4(scale_4.flatten(2))) + 
                             self.act_5(self.reduction_5(scale_5.flatten(2))))

        return  self.softmax(self.linear(scale_combination.flatten(1))) * 10 
               
               
               
               

#-- Testing Zone --#
if __name__== "__main__":
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create a dummy input x
    B, H, W, C, latent_dim = 300, 64, 256, 3, 16
    X = torch.randn(B,H,W,C).to(device)
    print("Input shape: ", X.shape)
    print("---")

    # initialize model
    model = SpaceEncoder(C, latent_dim).to(device)

    # run inference
    output = model(X)

    # test network output
    print("final network output shape: ", output.shape)
