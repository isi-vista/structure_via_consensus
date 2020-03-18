import math
import torch
import torch.nn as nn

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'constant':
                nn.init.constant_(m.weight,0.0)
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=1, std=0.02)
            nn.init.constant_(m.bias.data, 0)
            
    return init_fun


class FaceOccNet(nn.Module):
    def __init__(self, input_channels=3, n_classes=3, is_regularized=False):
        super().__init__()
        self.is_regularized=is_regularized
        self.model_enc = nn.Sequential(nn.ReflectionPad2d(1),
                                       nn.Conv2d(input_channels, 64, 3, 1, 0),
                                       nn.ELU(inplace=True),
                                       
                                       nn.ReflectionPad2d(1),
                                       nn.Conv2d(64, 128, 3, 2, 0),
                                       nn.ELU(inplace=True),
                                       nn.BatchNorm2d(128),
                                       
                                       nn.ReflectionPad2d(1),
                                       nn.Conv2d(128, 128, 3, 1, 0),
                                       nn.ELU(inplace=True),
                                       nn.BatchNorm2d(128),
                                       
                                       nn.ReflectionPad2d(1),
                                       nn.Conv2d(128, 128, 3, 1, 0),
                                       nn.ELU(inplace=True),
                                       nn.BatchNorm2d(128),
                                       
                                       nn.ReflectionPad2d(1),
                                       nn.Conv2d(128, 256, 3, 2, 0),
                                       nn.ELU(inplace=True),
                                       nn.BatchNorm2d(256))
        
        self.model_dilation = nn.Sequential(nn.ReflectionPad2d(4),
                                            nn.Conv2d(256, 256, 3, 1, 0, dilation=4),
                                            nn.ELU(inplace=True),
                                            nn.BatchNorm2d(256),

                                            nn.ReflectionPad2d(3),
                                            nn.Conv2d(256, 256, 3, 1, 0, dilation=3),
                                            nn.ELU(inplace=True),
                                            nn.BatchNorm2d(256))

        self.model_dec = nn.Sequential(nn.ReflectionPad2d(1),
                                       nn.Conv2d(512, 512, 3, 1, 0),
                                       nn.ELU(inplace=True),
                                       nn.BatchNorm2d(512),

                                       # nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.PixelShuffle(2),

                                       nn.ReflectionPad2d(1),
                                       nn.Conv2d(128, 128, 3, 1, 0),
                                       nn.ELU(inplace=True),
                                       nn.BatchNorm2d(128),

                                       nn.ReflectionPad2d(1),
                                       nn.Conv2d(128, 128, 3, 1, 0),
                                       nn.ELU(inplace=True),
                                       nn.BatchNorm2d(128),

                                       # nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.PixelShuffle(2),

                                       nn.ReflectionPad2d(1),
                                       nn.Conv2d(32, 32, 3, 1, 0),
                                       nn.ELU(inplace=True),
                                       nn.ReflectionPad2d(1),
                                       nn.Conv2d(32, 32, 3, 1, 0),
                                       nn.ELU(inplace=True),
        )

        self.model_class = nn.Conv2d(32, n_classes, 3, 1, 1)
        ## Init
        self.model_enc.apply(weights_init('xavier'))
        self.model_dilation.apply(weights_init('xavier'))
        self.model_dec.apply(weights_init('xavier'))
        self.model_class.apply(weights_init('xavier'))
        self.drop = nn.Dropout(p=0.2)

        
    def forward(self, input):
        output = self.model_enc(input)
        output = torch.cat([output, self.model_dilation(output)], 1)
        if self.is_regularized:
            output = self.drop(output)        
        output = self.model_dec(output)
        return self.model_class(output), None

                
if __name__ == "__main__":
    print('testing faceoccnet encoder-decoder model')
    model = FaceOccNet(3, 3).cuda()
    img = torch.randn(1, 3, 128, 128).cuda()
    output = model(img)
    print("output_size:", output.size())