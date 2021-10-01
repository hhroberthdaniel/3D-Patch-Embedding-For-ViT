# 3D-Patch-Embedding-For-ViT
This repository adapts Vision Transformer for 3D Data, by just changing the patch embedding algorithm.

Original ViT implementation: https://github.com/lucidrains/vit-pytorch


# Requirements

pytorch


einops

# Usage

model = ViT3D(image_size=(64,64,64), patch_size=(8,8,8), num_classes=1, dim=1024, channels=1, heads=4, depth=6, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
# (batch_size, num_channels, depth, width, height)
x = torch.rand(2, 1, 64, 64, 64)

y = model(x)
print(y.shape) 
# [2, 1]

