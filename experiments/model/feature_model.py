import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from timm.models.vision_transformer import VisionTransformer, resize_pos_embed
from torch import Tensor
from torchvision.transforms import functional as TVF


class SimpleCNNFeatureExtractor(nn.Module):
    """
    Simple CNN feature extractor trained from scratch for grayscale gradient images.
    Outputs both spatial features (B, D, H, W) and global features (B, D).
    """
    def __init__(self, in_channels=1, feature_dim=384, image_size=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.in_channels = in_channels
        
        # Encoder: progressively downsamples while increasing channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling for cls token equivalent
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image
            
        Returns:
            spatial_features: (B, D, H, W) - upscaled spatial features
            global_features: (B, D) - global pooled features
        """
        B, C, H, W = x.shape
        
        # Get spatial features through encoder
        spatial_features = self.encoder(x)  # (B, D, H//16, W//16)
        
        # Get global features via adaptive pooling
        global_features = self.global_pool(spatial_features)  # (B, D, 1, 1)
        global_features = global_features.squeeze(-1).squeeze(-1)  # (B, D)
        
        # Upsample spatial features to original resolution
        spatial_features = F.interpolate(
            spatial_features, size=(H, W), mode='bilinear', align_corners=False
        )  # (B, D, H, W)
        
        return spatial_features, global_features
    
    def normalize(self, img: Tensor):
        """Normalize grayscale images to [-1, 1]"""
        return (img - 0.5) / 0.5
    
    def denormalize(self, img: Tensor):
        """Denormalize from [-1, 1] to [0, 1]"""
        return torch.clip(img * 0.5 + 0.5, 0, 1)


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

MODEL_URLS = {
    'vit_base_patch16_224_mae': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
    'vit_small_patch16_224_msn': 'https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar',
    'vit_large_patch7_224_msn': 'https://dl.fbaipublicfiles.com/msn/vitl7_200ep.pth.tar',
}

NORMALIZATION = {
    'vit_base_patch16_224_mae': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'vit_small_patch16_224_msn': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'vit_large_patch7_224_msn': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
}

MODEL_KWARGS = {
    'vit_base_patch16_224_mae': dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
    ), 
    'vit_small_patch16_224_msn': dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
    ),
    'vit_large_patch7_224_msn': dict(
        patch_size=7, embed_dim=1024, depth=24, num_heads=16,
    )
}


class FeatureModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self, 
        image_size: int = 224,
        model_name: str = 'vit_small_patch16_224_mae',
        global_pool: str = '',  # '' or 'token'
        use_grayscale_normalization: bool = False,  # Use simple 0.5 mean/std for grayscale
        use_cnn_extractor: bool = False,  # NEW: Use SimpleCNNFeatureExtractor instead of ViT
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.use_grayscale_normalization = use_grayscale_normalization
        self.use_cnn_extractor = use_cnn_extractor
        self.is_cnn = use_cnn_extractor

        # Identity
        if self.model_name == 'identity':
            return

        # NEW: Use SimpleCNNFeatureExtractor
        if use_cnn_extractor:
            self.model = SimpleCNNFeatureExtractor(
                in_channels=1, 
                feature_dim=384,  # Match ViT feature dim for compatibility
                image_size=image_size
            )
            self.feature_dim = 384
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)
            self.fc = nn.Identity()
            return

        # Original ViT path
        # Create model
        self.model = VisionTransformer(
            img_size=image_size, num_classes=0, global_pool=global_pool,
            **MODEL_KWARGS[model_name])

        # Model properties
        self.feature_dim = self.model.embed_dim
        
        # Set normalization based on grayscale flag
        if self.use_grayscale_normalization:
            # Use simple normalization for grayscale: mean=0.5, std=0.5 for all channels
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)
        else:
            # Use ImageNet normalization
            self.mean, self.std = NORMALIZATION[model_name]

        # # Modify MSN model with output head from training
        # if model_name.endswith('msn'):
        #     use_bn = True
        #     emb_dim = (192 if 'tiny' in model_name else 384 if 'small' in model_name else 
        #         768 if 'base' in model_name else 1024 if 'large' in model_name else 1280)
        #     hidden_dim = 2048
        #     output_dim = 256
        #     self.model.fc = None
        #     fc = OrderedDict([])
        #     fc['fc1'] = torch.nn.Linear(emb_dim, hidden_dim)
        #     if use_bn:
        #         fc['bn1'] = torch.nn.BatchNorm1d(hidden_dim)
        #     fc['gelu1'] = torch.nn.GELU()
        #     fc['fc2'] = torch.nn.Linear(hidden_dim, hidden_dim)
        #     if use_bn:
        #         fc['bn2'] = torch.nn.BatchNorm1d(hidden_dim)
        #     fc['gelu2'] = torch.nn.GELU()
        #     fc['fc3'] = torch.nn.Linear(hidden_dim, output_dim)
        #     self.model.fc = torch.nn.Sequential(fc)
        
        # Load pretrained checkpoint
        checkpoint = torch.hub.load_state_dict_from_url(MODEL_URLS[model_name])
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'target_encoder' in checkpoint:
            state_dict = checkpoint['target_encoder']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # NOTE: Comment the line below if using the projection head, uncomment if not using it
            # See https://github.com/facebookresearch/msn/blob/81cb855006f41cd993fbaad4b6a6efbb486488e6/src/msn_train.py#L490-L502
            # for more info about the projection head
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        else:
            raise NotImplementedError()
        state_dict['pos_embed'] = resize_pos_embed(state_dict['pos_embed'], self.model.pos_embed)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # # Modify MSN model with output head from training
        # if model_name.endswith('msn'):
        #     self.fc = self.model.fc
        #     del self.model.fc
        # else:
        #     self.fc = nn.Identity()
        
        # NOTE: I've disabled the whole projection head stuff for simplicity for now
        self.fc = nn.Identity()

    def denormalize(self, img: Tensor):
        img = TVF.normalize(img, mean=[-m/s for m, s in zip(self.mean, self.std)], std=[1/s for s in self.std])
        return torch.clip(img, 0, 1)

    def normalize(self, img: Tensor):
        return TVF.normalize(img, mean=self.mean, std=self.std)

    def forward(
        self, 
        x: Tensor, 
        return_type: str = 'features',
        return_upscaled_features: bool = True,
        return_projection_head_output: bool = False,
    ):
        """Normalizes the input `x` and runs it through `model` to obtain features"""
        assert return_type in {'cls_token', 'features', 'all'}

        # Identity
        if self.model_name == 'identity':
            return x
        
        B, C, H, W = x.shape
        
        # NEW: CNN path
        if self.use_cnn_extractor:
            x_norm = self.model.normalize(x)
            spatial_features, global_features = self.model(x_norm)
            # spatial_features: (B, D, H, W) already at original resolution
            # global_features: (B, D)
            
            if return_type == 'cls_token':
                return global_features
            elif return_type == 'features':
                return spatial_features
            else:  # 'all'
                return global_features, spatial_features
        
        # Original ViT path
        # Normalize and forward
        x = self.normalize(x)
        feats = self.model(x)

        # Reshape to image-like size
        if return_type in {'features', 'all'}:
            B, T, D = feats.shape
            assert math.sqrt(T - 1).is_integer()
            HW_down = int(math.sqrt(T - 1))  # subtract one for CLS token
            output_feats: Tensor = feats[:, 1:, :].reshape(B, HW_down, HW_down, D).permute(0, 3, 1, 2)  # (B, D, H_down, W_down)
            if return_upscaled_features:
                output_feats = F.interpolate(output_feats, size=(H, W), mode='bilinear',
                    align_corners=False)  # (B, D, H_orig, W_orig)

        # Head for MSN
        output_cls = feats[:, 0]
        if return_projection_head_output and return_type in {'cls_token', 'all'}:
            output_cls = self.fc(output_cls)
        
        # Return
        if return_type == 'cls_token':
            return output_cls
        elif return_type == 'features':
            return output_feats
        else:
            return output_cls, output_feats
