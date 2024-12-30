from collections import OrderedDict
from typing import Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    """
    瓶颈模块（Bottleneck Block），常用于 ResNet 等深度卷积神经网络中。

    参数:
        expansion (int): 扩展因子，用于控制第三个卷积层的输出通道数。默认为4。
    """
    # 第三个卷积层的输出通道数是 planes 的 expansion 倍
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """
        初始化 Bottleneck 模块。

        参数:
            inplanes (int): 输入特征图的通道数。
            planes (int): 第一个卷积层的输出通道数。
            stride (int): 卷积层的步幅，默认为1。
        """
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        # 第一个卷积层：1x1 卷积，用于减少通道数
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        # 批归一化
        self.bn1 = nn.BatchNorm2d(planes)
        # ReLU 激活函数，inplace=True 表示原地操作，节省内存
        self.relu1 = nn.ReLU(inplace=True)

        # 第二个卷积层：3x3 卷积，保持通道数不变
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        # 批归一化
        self.bn2 = nn.BatchNorm2d(planes)
        # ReLU 激活函数
        self.relu2 = nn.ReLU(inplace=True)

        # 平均池化层：如果步幅大于1，则在第二个卷积层后添加一个平均池化层，用于下采样
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 第三个卷积层：1x1 卷积，用于增加通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        # 批归一化
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # ReLU 激活函数
        self.relu3 = nn.ReLU(inplace=True)

        # 下采样层：如果步幅大于1 或 输入通道数不等于扩展后的通道数，则需要进行下采样
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            # 下采样层由平均池化层和 1x1 卷积层组成
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)), # 平均池化层，用于空间下采样
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), # 1x1 卷积层，用于调整通道数
                ("1", nn.BatchNorm2d(planes * self.expansion)) # 批归一化层
            ])) 

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        # 保存输入作为恒等映射（identity）
        identity = x

        # 第一个卷积块：1x1 卷积 -> 批归一化 -> ReLU
        out = self.relu1(self.bn1(self.conv1(x)))
        # 第二个卷积块：3x3 卷积 -> 批归一化 -> ReLU
        out = self.relu2(self.bn2(self.conv2(out)))
        # 平均池化层（如果步幅大于1）
        out = self.avgpool(out)
        # 第三个卷积块：1x1 卷积 -> 批归一化
        out = self.bn3(self.conv3(out))

        # 如果需要进行下采样，则对输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 将输出与下采样后的输入相加
        out += identity
        # 最后经过 ReLU 激活函数
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    """
    一个二维注意力池化层，用于替代传统的平均池化或最大池化。

    参数:
        spacial_dim (int): 空间维度的大小，通常是特征图的高度或宽度。
        embed_dim (int): 嵌入维度的大小。
        num_heads (int): 多头注意力机制中的头数。
        output_dim (int, 可选): 输出维度的大小。如果未指定，则默认为 embed_dim。
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # 位置编码：生成一个可学习的参数，大小为 (spacial_dim^2 + 1) x embed_dim
        # 除以 embed_dim 的平方根进行初始化
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # 线性变换层，用于生成查询 (q)、键 (k) 和值 (v)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 线性变换层，用于最终的输出
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        # 多头注意力的头数
        self.num_heads = num_heads

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (N, C, H, W)。

        返回:
            torch.Tensor: 输出张量，形状为 (N, output_dim)。
        """
        # 将输入张量从 (N, C, H, W) 展平为 (N, C, H*W)，然后转置为 (H*W, N, C)
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        # 在序列的开头添加一个平均池化的全局特征向量
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # 添加位置编码，位置编码的形状为 (HW+1) x 1 x C
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        # 使用多头注意力机制
        # query, key, value 的形状分别为 (1, N, C), (HW+1, N, C), (HW+1, N, C)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],  # 查询 (query) 为全局特征向量
            key=x,        # 键 (key) 为全局特征向量加上位置编码
            value=x,      # 值 (value) 为全局特征向量加上位置编码
            embed_dim_to_check=x.shape[-1],      # 嵌入维度
            num_heads=self.num_heads,            # 注意力头数
            q_proj_weight=self.q_proj.weight,    # 查询的投影权重
            k_proj_weight=self.k_proj.weight,    # 键的投影权重
            v_proj_weight=self.v_proj.weight,    # 值的投影权重
            in_proj_weight=None,                 # 输入投影权重（未使用）
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),  # 输入投影偏置
            bias_k=None,                         # 键的偏置（未使用）
            bias_v=None,                         # 值的偏置（未使用）
            add_zero_attn=False,                 # 是否添加零注意力
            dropout_p=0,                         # dropout 概率
            out_proj_weight=self.c_proj.weight,  # 输出投影权重
            out_proj_bias=self.c_proj.bias,      # 输出投影偏置
            use_separate_proj_weight=True,       # 是否使用独立的投影权重
            training=self.training,              # 是否在训练模式下
            need_weights=False                   # 是否返回注意力权重
        )
        # 去除多余的维度并返回结果
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    一个改进版的 ResNet 模型，与 torchvision 的 ResNet 相比，具有以下变化：
    - 现在有 3 个“主干”卷积层，而不是 1 个，使用平均池化而不是最大池化。
    - 执行抗锯齿的步进卷积，在步进大于 1 的卷积之前添加一个平均池化层。
    - 最终的池化层是一个 QKV 注意力层，而不是平均池化。
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        """
        初始化 ModifiedResNet 模型。

        参数:
            layers (list): 每个阶段的瓶颈模块数量。
            output_dim (int): 输出维度的大小。
            heads (int): 多头注意力机制中的头数。
            input_resolution (int, 可选): 输入图像的分辨率，默认为 224。
            width (int, 可选): 初始卷积层的通道数，默认为 64。
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # 3 层主干卷积
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # 残差层
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # ResNet 的特征维度
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """
        构建一个残差层。

        参数:
            planes (int): 瓶颈模块的通道数。
            blocks (int): 瓶颈模块的数量。
            stride (int, 可选): 第一个瓶颈模块的步幅，默认为 1。

        返回:
            nn.Sequential: 包含多个瓶颈模块的序列。
        """
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        # 将输入张量转换为与卷积层权重相同的类型
        x = x.type(self.conv1.weight.dtype)
        # 主干卷积
        x = stem(x)
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 注意力池化
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    """
    继承自 torch 的 LayerNorm 类，以支持半精度浮点数 (fp16) 处理。

    LayerNorm 是一种归一化方法，用于稳定和加速神经网络的训练过程。
    """

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过层归一化处理后的张量，保持原始数据类型。
        """
        # 保存输入张量的原始数据类型
        orig_type = x.dtype
        # 将输入张量转换为 float32 类型，以便进行层归一化计算
        ret = super().forward(x.type(torch.float32))
        # 将输出张量转换回原始数据类型
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    QuickGELU 激活函数，一种近似于 GELU (高斯误差线性单元) 的快速实现。
    """
    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过 QuickGELU 激活函数处理后的张量。
        """
        # QuickGELU 的计算公式: x * sigmoid(1.702 * x)
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    残差注意力块（Residual Attention Block），是 Transformer 模型的核心组件之一。

    它结合了多头自注意力机制和前馈神经网络，并通过残差连接和层归一化进行增强。
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """
        初始化残差注意力块。

        参数:
            d_model (int): 模型中每个输入和输出样本的维度。
            n_head (int): 多头注意力机制中的头数。
            attn_mask (torch.Tensor, 可选): 注意力掩码，用于遮蔽某些位置。
        """
        super().__init__()

        # 多头自注意力机制
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # 层归一化层，用于注意力机制输入
        self.ln_1 = LayerNorm(d_model)
        # 前馈神经网络，由线性层、QuickGELU 激活函数和另一个线性层组成
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # 第一个线性层，将维度扩展 4 倍
            ("gelu", QuickGELU()),  # QuickGELU 激活函数
            ("c_proj", nn.Linear(d_model * 4, d_model))  # 第二个线性层，将维度恢复为原始大小
        ]))
        # 层归一化层，用于前馈神经网络输入
        self.ln_2 = LayerNorm(d_model)
        # 注意力掩码
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        """
        应用多头自注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 (L, N, E)，其中 L 是序列长度，N 是批量大小，E 是嵌入维度。

        返回:
            torch.Tensor: 经过注意力机制处理后的张量，形状与输入相同。
        """
        # 将注意力掩码转换为与输入张量相同的设备和数据类型
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 应用多头自注意力机制，need_weights=False 表示不返回注意力权重
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过残差注意力块处理后的张量。
        """
        # 第一个残差连接：注意力机制 + 层归一化
        x = x + self.attention(self.ln_1(x))
        # 第二个残差连接：前馈神经网络 + 层归一化
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer 模型，由多个残差注意力块组成。

    参数:
        width (int): 模型中每个输入和输出样本的维度。
        layers (int): 残差注意力块的数量。
        heads (int): 多头注意力机制中的头数。
        attn_mask (torch.Tensor, 可选): 注意力掩码，用于遮蔽某些位置。
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """
        初始化 Transformer 模型。

        参数:
            width (int): 模型中每个输入和输出样本的维度。
            layers (int): 残差注意力块的数量。
            heads (int): 多头注意力机制中的头数。
            attn_mask (torch.Tensor, 可选): 注意力掩码，用于遮蔽某些位置。
        """
        super().__init__()
        self.width = width
        self.layers = layers
        # 构建多个残差注意力块，并使用 Sequential 容器进行组合
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过 Transformer 模型处理后的张量。
        """
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """
    视觉 Transformer（Vision Transformer, ViT）模型，用于图像分类任务。

    ViT 将输入图像分割成固定大小的 patch块（patches），然后通过 Transformer 编码器进行处理。
    """
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        """
        初始化视觉 Transformer 模型。

        参数:
            input_resolution (int): 输入图像的分辨率（高度或宽度）。
            patch_size (int): 每个补丁的尺寸（高度或宽度）。
            width (int): Transformer 编码器的隐藏维度。
            layers (int): Transformer 编码器的层数。
            heads (int): 多头注意力机制中的头数。
            output_dim (int): 输出特征的维度。
        """
        super().__init__()
        # 输入图像的分辨率
        self.input_resolution = input_resolution
        # 输出特征的维度
        self.output_dim = output_dim
        # 第一个卷积层：将输入图像分割成补丁，并线性投影到隐藏维度
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # 缩放因子，用于初始化类嵌入和位置嵌入
        scale = width ** -0.5
        # 类嵌入（Class Token）：一个可学习的参数，用于表示图像的类别
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # 位置嵌入（Positional Embedding）：一个可学习的参数，用于编码每个补丁的位置信息
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        # 层归一化层，用于预处理输入
        self.ln_pre = LayerNorm(width)

        # Transformer 编码器
        self.transformer = Transformer(width, layers, heads)

        # 层归一化层，用于 Transformer 编码器输出
        self.ln_post = LayerNorm(width)
        # 投影矩阵：将 Transformer 编码器的输出投影到输出维度
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (N, C, H, W)。

        返回:
            torch.Tensor: 输出特征张量，形状为 (N, output_dim)。
        """
        # 通过卷积层将输入图像分割成补丁，并线性投影到隐藏维度
        # 输出形状为 [N, width, grid, grid]，其中 grid = input_resolution // patch_size
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        # 将补丁张量展平为形状 [N, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]

        # 转置张量形状为 [N, grid ** 2, width]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # 添加类嵌入和位置嵌入
        # 类嵌入被添加到每个样本的第一个位置
        # 形状为 [N, grid ** 2 + 1, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
       
        # 添加位置嵌入
        x = x + self.positional_embedding.to(x.dtype)
        # 层归一化预处理
        x = self.ln_pre(x)

        # 转置张量形状为 [grid ** 2 + 1, N, width]，以适应 Transformer 的输入格式
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 通过 Transformer 编码器处理
        x = self.transformer(x)
        # 转置张量形状回 [N, grid ** 2 + 1, width]
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 层归一化后处理，只取类嵌入对应的输出
        x = self.ln_post(x[:, 0, :])

        # 如果有投影矩阵，则将输出投影到输出维度
        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    """
    CLIP 模型，实现了图像和文本的联合编码，并通过对比学习进行训练。

    CLIP 模型由两个主要部分组成：
    1. 视觉编码器（Visual Encoder）：用于编码图像。
    2. 文本编码器（Text Encoder）：用于编码文本。

    参数:
        embed_dim (int): 
            嵌入维度，用于视觉编码器和文本编码器的输出维度。

        # 视觉部分参数
        image_resolution (int): 
            输入图像的分辨率（高度或宽度），例如 224。
        vision_layers (Union[Tuple[int, int, int, int], int]): 
            视觉编码器的层数。如果使用 ResNet，则为一个包含四个整数的元组，分别表示每个阶段的层数。
            如果使用 Vision Transformer，则为一个整数，表示 Transformer 编码器的层数。
        vision_width (int): 
            视觉编码器的宽度（通道数），例如 768。
        vision_patch_size (int): 
            ViT 中每个补丁的尺寸（高度或宽度），例如 32。

        # 文本部分参数
        context_length (int): 
            文本序列的最大长度，例如 77。
        vocab_size (int): 
            词汇表大小，例如 49408。
        transformer_width (int): 
            Transformer 编码器的宽度（嵌入维度），例如 512。
        transformer_heads (int): 
            Transformer 编码器的多头注意力头数，例如 8。
        transformer_layers (int): 
            Transformer 编码器的层数，例如 12。
    """
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        # 文本序列的最大长度
        self.context_length = context_length

        # 初始化视觉编码器
        if isinstance(vision_layers, (tuple, list)):
            # 如果 vision_layers 是一个元组或列表，则使用 ModifiedResNet 作为视觉编码器
            # 计算多头注意力的头数
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,  # ResNet 层数
                output_dim=embed_dim,  # 输出维度
                heads=vision_heads,    # 多头注意力的头数
                input_resolution=image_resolution,  # 输入图像的分辨率
                width=vision_width     # 视觉编码器的宽度
            )
        else:
            # 如果 vision_layers 是一个整数，则使用 VisionTransformer 作为视觉编码器
            # 计算多头注意力的头数
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,  # 输入图像的分辨率
                patch_size=vision_patch_size,       # ViT 中patch的大小
                width=vision_width,                 # 视觉编码器的宽度
                layers=vision_layers,               # ViT 层数
                heads=vision_heads,                 # 多头注意力的头数
                output_dim=embed_dim                # 输出维度
            )

        # 初始化文本编码器
        self.transformer = Transformer(
            width=transformer_width,    # Transformer 编码器的宽度
            layers=transformer_layers,  # Transformer 编码器的层数
            heads=transformer_heads,    # 多头注意力的头数
            attn_mask=self.build_attention_mask()  # 构建注意力掩码
        )

        # 词汇表大小
        self.vocab_size = vocab_size
        # 词嵌入层，将词汇表中的每个词映射到 Transformer 编码器的输入维度
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # 位置嵌入，可学习的参数，用于编码每个词的位置信息
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # 层归一化层，用于 Transformer 编码器的最终输出
        self.ln_final = LayerNorm(transformer_width)

        # 文本投影矩阵，将 Transformer 编码器的输出投影到嵌入维度
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # 对数尺度参数，用于调整相似度计算时的尺度
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 初始化模型参数
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        初始化模型参数。
        """
        # 初始化词嵌入层权重，使用标准差为0.02的正态分布
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # 初始化位置嵌入，使用标准差为0.01的正态分布
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            # 如果视觉编码器是 ModifiedResNet，则初始化特定参数
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                # 初始化注意力池化层的 q_proj, k_proj, v_proj, c_proj 权重
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            # 初始化 ResNet 残差块的最后一个批归一化层的权重为0
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # 初始化 Transformer 编码器的投影和注意力权重
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            # 初始化多头注意力的 q_proj, k_proj, v_proj 权重
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            # 初始化多头注意力的 out_proj 权重
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            # 初始化前馈神经网络的 c_fc 权重
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            # 初始化前馈神经网络的 c_proj 权重
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 初始化文本投影矩阵
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """
        构建注意力掩码，实现因果注意力。

        返回:
            torch.Tensor: 注意力掩码矩阵。
        """
        # 创建一个形状为 (context_length, context_length) 的空张量
        mask = torch.empty(self.context_length, self.context_length)
        # 用 -inf 填充整个张量
        mask.fill_(float("-inf"))
        # 将下三角部分设为0，实现因果注意力
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        """
        获取视觉编码器的权重数据类型。

        返回:
            torch.dtype: 视觉编码器权重的数据类型。
        """
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """
        对输入图像进行编码。

        参数:
            image (torch.Tensor): 输入图像张量。

        返回:
            torch.Tensor: 编码后的图像特征。
        """
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """
        对输入文本进行编码。

        参数:
            text (torch.Tensor): 输入文本张量。

        返回:
            torch.Tensor: 编码后的文本特征。
        """
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 取 eot_embedding 的特征（eot_token 是每个序列中最大的数字）
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        """
        前向传播函数，计算图像和文本的相似度。

        参数:
            image (torch.Tensor): 输入图像张量。
            text (torch.Tensor): 输入文本张量。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 图像到文本和文本到图像的相似度对数。
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        # 归一化特征
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        # 计算余弦相似度作为对数
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    """
    将适用的模型参数转换为半精度（fp16）。

    参数:
        model (nn.Module): 需要转换权重的模型。
    """

    def _convert_weights_to_fp16(l):
        """
        递归地将层中的权重转换为半精度。

        参数:
            layer (nn.Module): 需要转换权重的层。
        """
        # 如果层是卷积层（1D 或 2D）或全连接层，则将其权重和偏置转换为半精度
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # 将权重转换为半精度
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                # 将偏置转换为半精度
                l.bias.data = l.bias.data.half()

        # 如果层是多头注意力机制，则将其所有相关权重转换为半精度
        if isinstance(l, nn.MultiheadAttention):
            # 输入、查询、键、值投影权重
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                # 获取属性
                tensor = getattr(l, attr)
                if tensor is not None:
                    # 将属性转换为半精度
                    tensor.data = tensor.data.half()

        # 如果层有 "text_projection" 或 "proj" 属性，则将其转换为半精度
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    # 将属性转换为半精度
                    attr.data = attr.data.half()

    # 递归地应用转换函数到模型的所有层
    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    """
    根据状态字典构建 CLIP 模型。

    参数:
        state_dict (Dict[str, torch.Tensor]): 模型的预训练状态字典。

    返回:
        nn.Module: 加载了预训练权重的 CLIP 模型。
    """
    # 判断状态字典中是否存在 "visual.proj" 键，以确定使用哪种视觉编码器
    vit = "visual.proj" in state_dict

    if vit:
        # 如果使用 Vision Transformer (ViT) 作为视觉编码器，则从状态字典中提取相关参数
        # 视觉编码器的宽度
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        # ViT 层数
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        # patch大小
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        # 网格大小
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        # 图像分辨率
        image_resolution = vision_patch_size * grid_size
    else:
        # 如果使用 ModifiedResNet 作为视觉编码器，则从状态字典中提取相关参数
        # 计算每个阶段的层数
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        # 视觉编码器的层数
        vision_layers = tuple(counts)
        # 视觉编码器的宽度
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        # 输出宽度
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        # patch大小（未使用）
        vision_patch_size = None
        # 确保网格大小正确
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        # 图像分辨率
        image_resolution = output_width * 32
    
    # 从状态字典中提取其他参数
    # 嵌入维度
    embed_dim = state_dict["text_projection"].shape[1]
    # 上下文长度
    context_length = state_dict["positional_embedding"].shape[0]
    # 词汇表大小
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    # Transformer 编码器的宽度
    transformer_width = state_dict["ln_final.weight"].shape[0]
    # Transformer 编码器的多头注意力头数
    transformer_heads = transformer_width // 64
    # Transformer 编码器的层数
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    
    # 构建 CLIP 模型
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # 删除状态字典中不需要的键
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # 将模型权重转换为半精度
    convert_weights(model)
    # 加载状态字典到模型中
    model.load_state_dict(state_dict)
    # 设置模型为评估模式
    return model.eval()
