
import torch
from torch import nn

class UnetEncoderBlock(nn.Module):
   
    def __init__(self,inp_channel,out_channel):
        '''
        EncoderBlock of the Unet
        
        Args: inp_channel:int
              out_channel:int
        
        In the Unet setting, out_channel = 2*inp_channel.
        Eg, Input shape: (N,C,W,H), out shape: (N,2*C,W,H)
        
        '''
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(inp_channel,out_channel,kernel_size=(3,3)),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_channel),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,3)),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_channel)
                                  )
                                     
    def forward(self,x):
        
        x = self.encoder(x)
        
        return x
    
class UnetDecoderBlock(nn.Module):
    
    def __init__(self,inp_channel,out_channel):
        '''
        DecoderBlock of the Unet
        
        Args: inp_channel:int
              out_channel:int
        
        In the Unet setting, 2* out_channel = inp_channel.
        
        Eg, Input shape: (N,C,W,H), out shape: (N,C/4,W,H)
        
        Note that the output is 4 times smaller than the input because of the concatenation of feature map 
        from the input and the previous smaller. The first Conv2d layer halfs the input channel. 
        The upsampling ConvTranspose2d halfs the number of channel again.
        
        '''
        super().__init__()
        self.decoder = nn.Sequential(nn.Conv2d(inp_channel,out_channel,kernel_size=(3,3)),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_channel),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,3)),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ConvTranspose2d(out_channel,int(out_channel/2),kernel_size=(2,2),stride=2)
                                  )
    def forward(self,x):
        
        x = self.decoder(x)
        
        return x
        
class Bottlenecklayer(nn.Module):
    
    
    
    def __init__(self,inp_channel):
        '''
        Args: inp_channel: int
        
        The bottleneck layer is the middle layer in Unet where the width and height of feature map is the smallest,
        hence the name bottleneck.
        
        '''
        super().__init__()
        self.bottleneck = nn.Sequential(
                                        nn.Conv2d(inp_channel,inp_channel,kernel_size = (3,3)),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(inp_channel),
                                        nn.Conv2d(inp_channel,inp_channel,kernel_size = (3,3)),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(inp_channel),
                                        nn.ConvTranspose2d(inp_channel,inp_channel,kernel_size=2,stride=2)
                                       )
    def forward(self,x):
        
        return self.bottleneck(x)
                                        
                                                 

class Unet(nn.Module):
   
    def __init__(self):
        '''
    
        Implementation of Unet from described in the U-Net: Convolutional Networks for Biomedical Image Segmentation:
        https://arxiv.org/pdf/1505.04597.pdf

        This model was originally created for image segmentation. The only difference beteween this model and the original 
        model is we added a classification head at the end to do classification.
        
        '''
        
        super().__init__()
        
        self.encoder1 = UnetEncoderBlock(3,64)
        self.encoder2 = UnetEncoderBlock(64,128)
        self.encoder3 = UnetEncoderBlock(128,256)
        self.encoder4 = UnetEncoderBlock(256,512)
        
        self.bottlenecklayer = Bottlenecklayer(512)
        
        
        self.decoder3 = UnetDecoderBlock(1024,512)
        self.decoder2 = UnetDecoderBlock(512,256)
        self.decoder1 = UnetDecoderBlock(256,128)
        self.down_sample = nn.MaxPool2d(kernel_size=2)
        
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
        Args: encoder_feature :torch.tensor, shape [B,C,W_e,H_e]
            : decoder_feature :torch.tensor, shape [B,C,W_d,W_d]
            
        Returns:
            combined: torch.tensor, shape [B,2*C,W_d,W_d]
        
        
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
            
        combined = torch.cat([cropped,decoder_feature],dim=1) ## concat the 2 featuremap
        
        return combined
            
       
    def forward(self,x):
        
        encoder1_output = self.encoder1(x)
        encoder2_output = self.encoder2(self.down_sample(encoder1_output))
        encoder3_output = self.encoder3(self.down_sample(encoder2_output))
        encoder4_output = self.encoder4(self.down_sample(encoder3_output))
        bottleneck_output = self.bottlenecklayer(self.down_sample(encoder4_output))
       
        ## The combined feature map for encoder4. Use it and feed it into the decoder. The logic is the same
        ## for the other decoder.
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
        
        
        
       
        
        
