import torchvision.transforms as transforms
import random
import torch


def build_transform(config):
    """
    根据配置文件构建 transform 流水线
    config 为一个字典，包含你需要的增强及参数
    例如:
        config = {
            'crop': {'type': 'RandomResizedCrop', 'size': 224},
            'flip': {'type': 'RandomHorizontalFlip', 'p': 0.5},
            'color_jitter': {'brightness': 0.8, 'contrast': 0.8, 'saturation': 0.8, 'hue': 0.2, 'p': 0.8},
            'grayscale': {'p': 0.2},
            'gaussian_blur': {'kernel_size': 23, 'p': 0.5}
        }
    """
    transform_list = []
    
    # # Crop
    if 'crop' in config:
        crop_config = config['crop']
        crop_type = crop_config.get('type', 'RandomResizedCrop')
        size = crop_config.get('size', 224)
        if crop_type == 'RandomResizedCrop':
            transform_list.append(transforms.RandomResizedCrop(size))
        elif crop_type == 'RandomCrop':
            transform_list.append(transforms.RandomCrop(size))
        else:
            raise ValueError(f"Unknown crop type: {crop_type}")

    # Flip
    if 'flip' in config:
        flip_config = config['flip']
        p = flip_config.get('p', 0.5)
        transform_list.append(transforms.RandomHorizontalFlip(p=p))

    # Color Jitter
    if 'color_jitter' in config:
        cj_config = config['color_jitter']
        brightness = cj_config.get('brightness', 0.8)
        contrast   = cj_config.get('contrast', 0.8)
        saturation = cj_config.get('saturation', 0.8)
        hue        = cj_config.get('hue', 0.2)
        p          = cj_config.get('p', 0.8)
        color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                              saturation=saturation, hue=hue)
        transform_list.append(transforms.RandomApply([color_jitter], p=p))

    # Random Grayscale
    if 'grayscale' in config:
        gray_config = config['grayscale']
        p = gray_config.get('p', 0.2)
        transform_list.append(transforms.RandomGrayscale(p=p))

    # Gaussian Blur
    if 'gaussian_blur' in config:
        gb_config = config['gaussian_blur']
        kernel_size = gb_config.get('kernel_size', 23)
        p = gb_config.get('p', 0.5)
        gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size)
        transform_list.append(transforms.RandomApply([gaussian_blur], p=p))


    # Normalize (如果需要)
    if 'normalize' in config:
        norm_config = config['normalize']
        mean = norm_config.get('mean', [0.485, 0.456, 0.406])
        std  = norm_config.get('std', [0.229, 0.224, 0.225])
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)
