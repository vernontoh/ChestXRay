import torch.nn as nn
import torch



# XvT with dropout layers
class XvtConvEmbeddings(nn.Module):
    def __init__(self, patch_size, num_channels, embed_dim, stride, padding, drop_rate):
        super().__init__()
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.dropout = nn.Dropout(drop_rate)
        self.normalization = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values):
        pixel_values = self.projection(pixel_values)
        batch_size, num_channels, height, width = pixel_values.shape
        hidden_size = height * width
        # rearrange "b c h w -> b (h w) c"
        pixel_values = pixel_values.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        if self.normalization:
            pixel_values = self.normalization(pixel_values)
        # rearrange "b (h w) c" -> b c h w"
        pixel_values = pixel_values.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        pixel_values = self.dropout(pixel_values)

        return pixel_values


class XvtSelfAttentionProjection(nn.Module):
    def __init__(self, embed_dim, patch_size, padding, stride):
        super().__init__()

        self.convolution = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, padding=padding, stride=stride, bias=False, groups=embed_dim)
        self.normalization = nn.BatchNorm2d(embed_dim)

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)

        batch_size, num_channels, height, width = hidden_state.shape
        hidden_size = height * width
        # rearrange " b c h w -> b (h w) c"
        hidden_state = hidden_state.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        return hidden_state


class XvtSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        kernel_size,
        padding,
        stride,
        num_heads,
        drop_rate
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        self.num_heads = num_heads

        self.conv_proj_query = XvtSelfAttentionProjection(embed_dim, kernel_size, padding, stride)
        self.conv_proj_key = XvtSelfAttentionProjection(embed_dim, kernel_size, padding, stride)
        self.conv_proj_value = XvtSelfAttentionProjection(embed_dim, kernel_size, padding, stride)

        self.proj_query = nn.Linear(embed_dim, embed_dim)
        self.proj_key = nn.Linear(embed_dim, embed_dim)
        self.proj_value = nn.Linear(embed_dim, embed_dim)

        self.dropout_attention = nn.Dropout(drop_rate)
        self.dropout_output = nn.Dropout(drop_rate)

        self.output = nn.Linear(embed_dim, embed_dim)

    def rearrange_for_multi_head_attention(self, hidden_state):
        batch_size, hidden_size, _ = hidden_state.shape
        head_dim = self.embed_dim // self.num_heads
        # rearrange 'b t (h d) -> b h t d'
        return hidden_state.view(batch_size, hidden_size, self.num_heads, head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_state, height, width):
        batch_size, hidden_size, num_channels = hidden_state.shape

        # rearrange "b (h w) c -> b c h w"
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        query = self.conv_proj_query(hidden_state)
        key = self.conv_proj_key(hidden_state)
        value = self.conv_proj_value(hidden_state)

        head_dim = self.embed_dim // self.num_heads

        query = self.rearrange_for_multi_head_attention(self.proj_query(query))
        key = self.rearrange_for_multi_head_attention(self.proj_key(key))
        value = self.rearrange_for_multi_head_attention(self.proj_value(value))
        
        attention_score = torch.einsum("bhlk,bhtk->bhlt", [query, key]) * self.scale
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
        attention_probs = self.dropout_attention(attention_probs)

        context = torch.einsum("bhlt,bhtv->bhlv", [attention_probs, value])
        # rearrange"b h t d -> b t (h d)"
        _, _, hidden_size, _ = context.shape
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, hidden_size, self.num_heads * head_dim)
        attention_output = self.output(context)
        attention_output = self.dropout_output(attention_output)
        return attention_output


class XvtIntermediate(nn.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super().__init__()
        self.dense = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.activation = nn.GELU()

    def forward(self, hidden_state):
        hidden_state = self.dense(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class XvtOutput(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, drop_rate):
        super().__init__()
        self.dense = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, hidden_state, input_tensor):
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = hidden_state + input_tensor
        return hidden_state


class XvtLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        kernel_size,
        padding,
        stride,
        mlp_ratio,
        num_heads,
        drop_rate
    ):
        super().__init__()
        self.attention = XvtSelfAttention(
            embed_dim,
            kernel_size,
            padding,
            stride,
            num_heads,
            drop_rate
        )

        self.intermediate = XvtIntermediate(embed_dim, mlp_ratio)
        self.output = XvtOutput(embed_dim, mlp_ratio, drop_rate)
        self.layernorm_before = nn.LayerNorm(embed_dim)
        self.layernorm_after = nn.LayerNorm(embed_dim)


    def forward(self, hidden_state, height, width):
        self_attention_output = self.attention(self.layernorm_before(hidden_state), height, width)

        # first residual connection
        hidden_state = self_attention_output + hidden_state

        layer_output = self.layernorm_after(hidden_state)
        layer_output = self.intermediate(layer_output)

        # second resudiaul connection
        layer_output = self.output(layer_output, hidden_state)
        return layer_output
    


class XvtEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(
            *[
                XvtLayer(
                    embed_dim=config.embed_dim,
                    kernel_size=config.kernel_qkv,
                    padding=config.padding_qkv,
                    stride=config.stride_qkv,
                    mlp_ratio=config.mlp_ratio,
                    num_heads=config.num_heads,
                    drop_rate=config.drop_rate
                )
            for _ in range(config.num_layers)
            ]
        )

    def forward(self, hidden_state):

        batch_size, num_channels, height, width = hidden_state.shape
        # rearrange b c h w -> b (h w) c"
        hidden_state = hidden_state.view(batch_size, num_channels, height * width).permute(0, 2, 1)

        for layer in self.layers:
            layer_outputs = layer(hidden_state, height, width)
            hidden_state = layer_outputs

        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        return hidden_state


class XvtForImageClassification(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = XvtConvEmbeddings(
            patch_size=config.patch_size, 
            num_channels=config.num_channels,
            embed_dim=config.embed_dim, 
            stride=config.patch_stride, 
            padding=config.patch_padding,
            drop_rate=config.drop_rate
        )

        self.encoder = XvtEncoder(config)
        self.layernorm = nn.LayerNorm(config.embed_dim)

        # classification head
        self.classifier = nn.Linear(config.embed_dim, config.num_labels)

    def forward(self, pixel_values):

        hidden_state = self.embedding(pixel_values)

        encoder_output = self.encoder(hidden_state)
 
        batch_size, num_channels, height, width = encoder_output.shape
        # rearrange "b c h w -> b (h w) c"
        encoder_output = encoder_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        encoder_output = self.layernorm(encoder_output)
        sequence_output_mean = encoder_output.mean(dim=1)

        logits = self.classifier(sequence_output_mean)
        return logits