import torch
from torch import nn

class SpaceDecoder(nn.Module):

    def __init__(self, channel_out, latent_dim):
        super().__init__()
        
        # constants
        self.height = 64
        self.width = 256
        self.middle_layers = 16
        self.channel_out = channel_out

        #-- normalize data
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        
        #-- expansion to latent_dim
        self.linear = nn.Linear(latent_dim, latent_dim*self.middle_layers)


        #-- scale recomposition
        self.deconv_1 = nn.ConvTranspose2d(latent_dim, channel_out, kernel_size=5, stride=2, padding=1)
        self.act_1 = nn.LeakyReLU(inplace=True)
        #--
        self.deconv_2 = nn.ConvTranspose2d(channel_out, channel_out, kernel_size=5, stride=2, padding=2)
        self.act_2 = nn.LeakyReLU(inplace=True)
        #--
        self.deconv_3 = nn.ConvTranspose2d(channel_out, channel_out, kernel_size=5, stride=2, padding=2)
        self.act_3 = nn.LeakyReLU(inplace=True)
        #--
        self.deconv_4 = nn.ConvTranspose2d(channel_out, channel_out, kernel_size=5, stride=2, padding=2)
        self.act_4 = nn.LeakyReLU(inplace=True)
        #--
        self.deconv_5 = nn.ConvTranspose2d(channel_out, channel_out, kernel_size=5, stride=2, padding=2)
        self.act_5 = nn.LeakyReLU(inplace=True)
        
        #-- scale expansion
        self.expansion_1 = nn.Linear(5*17, (64*256)//4)
        self.act_6 = nn.LeakyReLU(inplace=True)

        self.expansion_2 = nn.Linear(9*33,  (64*256)//4)
        self.act_7 = nn.LeakyReLU(inplace=True)

        self.expansion_3 = nn.Linear(17*65,  (64*256)//4)
        self.act_8 = nn.LeakyReLU(inplace=True)

        self.expansion_4 = nn.Linear(33*129,  (64*256)//4)
        self.act_9 = nn.LeakyReLU(inplace=True)

        self.expansion_5 = nn.Linear(65*257,  (64*256//4)) # before we //4 and expand with a last comb)
        self.act_10 = nn.LeakyReLU(inplace=True)
        #-- scale combination
        # self.comb_conv = nn.Sequential(
        #                         nn.ConvTranspose2d(channel_out, channel_out, kernel_size=5, stride=2, padding=1),
        #                         nn.LeakyReLU(inplace=True),
        #                         nn.ConvTranspose2d(channel_out, channel_out, kernel_size=5, stride=1, padding=2),
        #                         nn.LeakyReLU(inplace=True),
        #                         nn.ConvTranspose2d(channel_out, channel_out, kernel_size=5, stride=1, padding=2),
                            #) 
        #self.comb_conv = nn.ConvTranspose2d(channel_out, channel_out, kernel_size=5, stride=2, padding=1)
        #self.act_11 = nn.LeakyReLU(inplace=True)
        #self.comb_conv2 = nn.ConvTranspose2d(channel_out, channel_out, kernel_size=5, stride=1, padding=2)
        self.comb_layer =  nn.Linear((64*256)//4,64*256)
    def forward(self,X):
        batch_size, latent_dim = X.shape

        #X = self.batch_norm(X)

        #-- expand latent space
        X = self.linear(X).view(batch_size, self.middle_layers, self.height//32, self.width//32) 
        
        #-- scale creation
        scale_1 = self.act_1(self.deconv_1(X))
        scale_2 = self.act_2(self.deconv_2((scale_1)))
        scale_3 = self.act_3(self.deconv_3((scale_2)))
        scale_4 = self.act_4(self.deconv_4((scale_3)))
        scale_5 = self.act_5(self.deconv_5((scale_4)))

        # renconstruct features at different scale
        scale_combination = (self.act_6(self.expansion_2(scale_2.flatten(2))) + 
                            self.act_7(self.expansion_2(scale_2.flatten(2))) +
                            self.act_8(self.expansion_3(scale_3.flatten(2))) +
                            self.act_9(self.expansion_4(scale_4.flatten(2))) +
                            self.act_10(self.expansion_5(scale_5.flatten(2))))#.view(batch_size, self.channel_out, self.height, self.width)
        
        return self.comb_layer(scale_combination).view(batch_size,self.channel_out,self.height, self.width).permute(0,2,3,1)#self.comb_conv(scale_combination)[:,:,:self.height, :self.width].permute(0,2,3,1)       

#-- Testing Zone --#
if __name__== "__main__":
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create a dummy input x
    B, H, W, C, latent_dim = 300, 64, 256, 3, 16
    X = torch.randn(B, latent_dim).to(device)

    # initialize model
    model = SpaceDecoder(C, latent_dim).to(device)

    # run inference
    output = model(X)

    # test network output
    print("Network Output Shape: ", output.shape)

