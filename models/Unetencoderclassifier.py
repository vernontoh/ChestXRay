
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
                                   nn.Dropout(p=0.1),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,3)),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_channel)
                                  )
                                     
    def forward(self,x):
        
        x = self.encoder(x)
        
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
                                        nn.BatchNorm2d(inp_channel)
                                       )
    def forward(self,x):
        
        return self.bottleneck(x)
                                        
                                                 

class UnetEncoderClassification(nn.Module):
   
    def __init__(self):
        '''
    
        Implementation of Unet from described in the U-Net: Convolutional Networks for Biomedical Image Segmentation:
        https://arxiv.org/pdf/1505.04597.pdf

        This model was originally created for image segmentation. The difference beteween this model and the original 
        model is we removed the decoder part and added a classification head at the end of the encoder to do classification.
        
        '''
        
        super().__init__()
        
        self.encoder1 = UnetEncoderBlock(3,64)
        self.encoder2 = UnetEncoderBlock(64,128)
        self.encoder3 = UnetEncoderBlock(128,256)
        self.encoder4 = UnetEncoderBlock(256,512)
        self.encoder5 = UnetEncoderBlock(512,1024)
        self.down_sample = nn.MaxPool2d(kernel_size=2)
        
        self.bottleneckconv = nn.Sequential(nn.Conv2d(1024,512,kernel_size=2),nn.ReLU(),nn.BatchNorm2d(512))
      
        
        #self.bottlenecklayer = Bottlenecklayer(1024)
        

   
        
        self.classification_head = nn.Sequential(
                                                 nn.Flatten(),
                                                 nn.Dropout(p=0.1),
                                                 nn.Linear(12800,14)
                                               
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
        #encoder1_cropped = crop_feature_map(encoder1_output,self.down_sample(encoder1_output))
        encoder2_output = self.encoder2(self.down_sample(encoder1_output))
        #encoder2_cropped = crop_feature_map(encoder2_output,self.down_sample(encoder2_output))
        encoder3_output = self.encoder3(self.down_sample(encoder2_output))
        encoder4_output = self.encoder4(self.down_sample(encoder3_output))
        encoder5_output = self.encoder5(self.down_sample(encoder4_output))
        bottleneck_out = self.bottleneckconv(encoder5_output)
      
        
        
        return self.classification_head(bottleneck_out)
      
        
        
        
       
        
        
