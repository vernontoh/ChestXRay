
class XvtConfig():
    r"""
    Args:
        num_labels (`int`, *optional*, defaults to `14`):
            The number of class labels.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of input channels.
        num_heads (`int`, *optional*, defaults to `1`):
            Number of attention heads for each attention layer.
        num_layers (`int`, *optional*, defaults to `1`):
            Number of layers in encoder block.
        patch_size (`int`, *optional*, defaults to `7`):
            The kernel size of patch embedding.
        patch_stride (`int`, *optional*, defaults to `4`):
            The stride size of patch embedding.
        patch_padding (`int`, *optional*, defaults to `2`):
            The padding size of patch embedding.
        embed_dim (`int`, *optional*, defaults to `64`):
            Dimension of convolution emedding.
        mlp_ratios (float`, *optional*, defaults to `4.0`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        kernel_qkv (`int`, *optional*, defaults to `3`):
            The kernel size for query, key and value in attention layer
        padding_qkv (`int`, *optional*, defaults to `1`):
            The padding size for key and value in attention layer
        stride_qkv (`int`, *optional*, defaults to `1`):
            The stride size for key and value in attention layer
    """
    def __init__(
        self,
	    num_labels=14,
        num_channels=3,
        num_heads=3,        #1,3,6
        num_layers=6,       #1,2,10
        patch_size=14,       #7,3,3
        patch_stride=14,     #4,2,2
        patch_padding=0,    #2,1,1
        embed_dim=192,       #64,192,384 
        mlp_ratio=4.0,      #4,4,4
        kernel_qkv=3,       #3 (fixed)
        padding_qkv=1,      #1 (fixed)
        stride_qkv=1,       #1 (fixed)
        drop_rate=0.3
   ):
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.kernel_qkv = kernel_qkv
        self.padding_qkv = padding_qkv
        self.stride_qkv = stride_qkv
        self.drop_rate = drop_rate

class XvtScheduler():
    def __init__(self,
                sched='cosine',
                warmup_epochs=2,
                warmup_lr=0.000001,
                min_lr=0.00001,
                cooldown_epochs=4,
                decay_rate=0.1,
                epochs=40
    ):
        self.sched = sched
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.cooldown_epochs = cooldown_epochs
        self.decay_rate = decay_rate
        self.epochs = epochs

        
        