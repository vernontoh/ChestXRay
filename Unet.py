
import torch
from torch import nn

class UnetEncoderBlock(nn.Module):
    
    def __init__(self,inp_channel,out_channel):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(inp_channel,out_channel,kernel_size=(3,3)),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,3)),
                                   nn.ReLU()
                                  )
                                     
    def forward(self,x):
        
        x = self.encoder(x)
        
        return x
    
class UnetDecoderBlock(nn.Module):
    
    def __init__(self,inp_channel,out_channel):
        super().__init__()
        self.decoder = nn.Sequential(nn.Conv2d(inp_channel,out_channel,kernel_size=(3,3)),
                                   nn.ReLU(),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,3)),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(out_channel,int(out_channel/2),kernel_size=(2,2),stride=2)
                                  )
    def forward(self,x):
        
        x = self.decoder(x)
        
        return x
        
class Bottlenecklayer(nn.Module):
    
    
    def __init__(self,inp_channel,out_channel):
        
        super().__init__()
        self.bottleneck = nn.Sequential(
                                        nn.Conv2d(inp_channel,out_channel,kernel_size = (3,3)),
                                        nn.ReLU(),
                                        nn.Conv2d(out_channel,out_channel,kernel_size = (3,3)),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(out_channel,out_channel,kernel_size=2,stride=2)
                                       )
    def forward(self,x):
        
        return self.bottleneck(x)
                                        
                                                 
        
        
        

class Unet(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.encoder1 = UnetEncoderBlock(3,64)
        self.encoder2 = UnetEncoderBlock(64,128)
        self.encoder3 = UnetEncoderBlock(128,256)
        self.encoder4 = UnetEncoderBlock(256,512)
        self.down_sample = nn.MaxPool2d(kernel_size=2)
        
        self.decoder3 = UnetDecoderBlock(1024,512)
        self.decoder2 = UnetDecoderBlock(512,256)
        self.decoder1 = UnetDecoderBlock(256,128)
        self.bottlenecklayer = Bottlenecklayer(512,512)
        
        self.decoder_feedforward = nn.Sequential(nn.Conv2d(128,64,kernel_size=3),
                                       nn.ReLU(),
                                       nn.Conv2d(64,64,kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv2d(64,2,kernel_size=1)
                                                )
        
        self.classification_head = nn.Sequential(nn.ReLU(),
                                                 nn.Flatten(),
                                                 nn.Linear(2592,14)
                                                )
    
    def crop_feature_map(self,encoder_feature,decoder_feature):
        '''
        Input dim: (B,C,W,H)
        
        In this case, the width of encoder feature is always greater than the decoder feature. To combine them,
        we crop the center of the encoder feature so that the 2 feature map are of the same shape.
        '''
        width_diff = encoder_feature.size(3)-decoder_feature.size(3) ## Get the width difference. We assume W=H
        
        if width_diff%2 == 0:
            start_dim = int(width_diff/2)
            
            cropped = encoder_feature[:,:,start_dim:-start_dim,start_dim:-start_dim]
        else:
            start_dim = int((width_diff-1)/2)
            cropped = encoder_feature[:,:,start_dim:-start_dim-1,start_dim:-start_dim-1]
            
        combined = torch.cat([cropped,decoder_feature],dim=1)
        
        return combined
            
       
    def forward(self,x):
        
        encoder1_output = self.encoder1(x)
        encoder2_output = self.encoder2(self.down_sample(encoder1_output))
        encoder3_output = self.encoder3(self.down_sample(encoder2_output))
        encoder4_output = self.encoder4(self.down_sample(encoder3_output))
        bottleneck_output = self.bottlenecklayer(self.down_sample(encoder4_output))
       
        
        concat_enc4 = self.crop_feature_map(encoder4_output,bottleneck_output)
        decoder3_output = self.decoder3(concat_enc4)
       
        concat_enc3 = self.crop_feature_map(encoder3_output,decoder3_output)
        
        decoder2_output = self.decoder2(concat_enc3)
        
        concat_enc2 = self.crop_feature_map(encoder2_output,decoder2_output)
        decoder1_output = self.decoder1(concat_enc2)
        concat_enc1 = self.crop_feature_map(encoder1_output,decoder1_output)
        Unet_output = self.decoder_feedforward(concat_enc1)
        
        ## Additional classification head added to do classification
        logits = self.classification_head(Unet_output)
        
        
        return logits
        
        
        
       
        
        
