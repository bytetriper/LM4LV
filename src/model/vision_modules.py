import requests
import torch
from PIL import Image, ImageFile
from header import *
from transformers import CLIPProcessor
from torch.nn import functional as F
from torchvision import transforms as T
from .CLIP import load as load_clip
from omegaconf import OmegaConf
from ldm.models.autoencoder import VQModel
from ldm.util import instantiate_from_config, get_obj_from_str
from transformers import AutoImageProcessor
from transformers import BeitForMaskedImageModeling
from transformers import ViTMAEModel,ViTImageProcessor,ViTMAEForPreTraining
from transformers import Dinov2Model
from dall_e.decoder import Decoder
from dall_e import unmap_pixels,load_model
from concurrent.futures import ThreadPoolExecutor
ImageFile.LOAD_TRUNCATED_IMAGES = True
image_type_mixin = Union[Image.Image, list[Image.Image], str, list[str],torch.Tensor]
text_type_mixin = Union[str, list[str]]
IM_SIZE = 224 # default to 224
def load_image(image_path,return_np,image_size:int = IM_SIZE, to_tensor:bool = False):
    """Load an image and return the path and size of the image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB').resize((image_size, image_size), Image.BICUBIC)
            if return_np:
                img = np.array(img).transpose(2, 0, 1)
                if to_tensor: # convert to tensor and normalize, only take effect when return_np is True
                    img = torch.Tensor(img).float()/255.0
            return (image_path,img)
    except Exception as e:
        raise Exception(f'Error loading image {image_path}: {e}')

def read_images_with_executor(folder_path:Union[str,list[str]], max_workers=4,im_size:int = IM_SIZE, recursive=False,return_np:bool=False, show_tqdm:bool = False):
    """Read all images in the given folder using ThreadPoolExecutor."""
    if isinstance(folder_path, str):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f'Folder not found: {folder_path}')
        valid_img_suffix = ['.png', '.jpg', '.jpeg']
        if recursive:
            image_paths = []
            for root, _, files in os.walk(folder_path):
                for f in files:
                    if f.lower().endswith(tuple(valid_img_suffix)):
                        image_paths.append(os.path.join(root, f))
        else:
            image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_paths = folder_path # a list of image paths
    im_sizes = [im_size] * len(image_paths)
    return_nps = [return_np]*len(image_paths)
    # use ThreadPoolExecutor to load images in parallel, with max_workers threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if show_tqdm:
            results = list(tqdm(executor.map(load_image, image_paths,return_nps,im_sizes), total=len(image_paths)))
        else:
            results = list(executor.map(load_image, image_paths,return_nps,im_sizes))
    # return a list of tuples, each containing the image path and the image
    return results
def count_parameters(model):
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters())
def load_and_transform_image_data(image_infos:image_type_mixin,im_size:int = 224)->list[Image.Image]:
    if image_infos is None:
        raise ValueError("image_infos is None")
    if not isinstance(image_infos, list) and not isinstance(image_infos, tuple):
        image_infos = [image_infos]
    image_ouputs = []
    for image_info in image_infos:
        if isinstance(image_info, Image.Image):
            image = image_info
        elif isinstance(image_info, str):
            if os.path.exists(image_info):
                image = Image.open(image_info)
            elif image_info.startswith("http://") or image_info.startswith("https://"):
                image = Image.open(requests.get(image_info, stream=True).raw)
            else:
                raise ValueError(f"Invalid image path: {image_info}")
        else:
            raise ValueError(f"Invalid image info: {image_info}")
        image = image.convert("RGB").resize((im_size, im_size), Image.BICUBIC)
        image_ouputs.append(image)
    return image_ouputs
def load_and_transform_text_data(text_infos: text_type_mixin) -> list[str]:
    if text_infos is None:
        raise ValueError("text_infos is None")
    if not isinstance(text_infos, list):
        text_infos = [text_infos]
    text_outputs = []
    for text_info in text_infos:
        if isinstance(text_info, str):
            text = text_info
        else:
            raise ValueError(f"Invalid text info: {text_info}")
        text_outputs.append(text)
    return text_outputs
def load_and_transform_data_mixin(data_infos: Union[str, list[str]], vision_type: str = 'image') -> list[Image.Image]:
    load_transform_method_str = f"load_and_transform_{vision_type}_data"
    assert load_transform_method_str in globals(), f"load_transform_method_str {load_transform_method_str} not found"
    load_transform_method = globals()[load_transform_method_str]
    return load_transform_method(data_infos)
def device_and_dtype(args: dict) -> tuple[torch.device, torch.dtype]:
    """
    Get the device and dtype
    :param args (dict): arguments
        {
            'device': str/torch.device,
            'dtype': str/torch.dtype
        }
    :return: device and dtype
    """
    device = args['device']
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, torch.device):
        device = device
    else:
        raise ValueError("device must be str or torch.device, but got {}".format(type(device)))
    dtype = args['dtype']
    if isinstance(dtype, str):
        dtype = get_obj_from_str(dtype)
    elif isinstance(dtype, torch.dtype):
        dtype = dtype
    else:
        raise ValueError("dtype must be str or torch.dtype, but got {}".format(type(dtype)))
    return device, dtype
def base_preprocess(images:image_type_mixin, image_mean:torch.Tensor, image_std:torch.Tensor, device:torch.device, dtype:torch.dtype,im_size:int = IM_SIZE)->torch.Tensor:
    """
    A general base preprocess function for reading images and preprocess them.
    If the images is a list of image paths, this function read the images in parallel.
    """
    if isinstance(images, torch.Tensor):
        image_tensors = images
    else:
        if (not isinstance(images, list)) and (not isinstance(images, tuple)):
                images = [images]
        # we assert the images is all str or torch.Tensor
        datatype = images[0]
        if isinstance(datatype, torch.Tensor):
            image_tensors = torch.stack(images)
        elif isinstance(datatype, np.ndarray):
            np_images = np.array(images) # create tensor from list of ndarray is slow, so convert them to a single ndarray first.
            image_tensors = torch.tensor(np_images,dtype= torch.float32) / 255.
        elif isinstance(datatype, str):
            np_images = read_images_with_executor(images, max_workers=8, return_np=True,im_size=im_size)
            names = [name for name, _ in np_images]
            # assert name to be equal to images
            assert names == images, "names must be equal to images, but got {} and {}".format(names, images)
            np_image = np.array([np_image for _, np_image in np_images])
            image_tensors = torch.tensor(np_image, dtype=torch.float32) / 255.
        else:
            raise ValueError("images must be list of np.ndarray or torch.Tensor, but got {}".format(type(datatype)))
    image_tensors = image_tensors.to(device).to(dtype)
    image_tensors = (image_tensors - image_mean) / image_std
    return image_tensors

"""-------------------------- Encoder Modules --------------------------"""
class BEiT_Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device, self.dtype = device_and_dtype(kwargs)
        ckpt_path = kwargs['ckpt_path']
        model = BeitForMaskedImageModeling.from_pretrained(ckpt_path)
        self.model = model.beit
        self.norm = model.layernorm
        self.model = self.model.to(self.dtype).to(self.device)
        self.model.eval()
        self.layer = kwargs.get('layer', -1) # used to specify the layer in the vision transformer to extract features
        self.im_size = kwargs.get('im_size', 224)
        self.image_mean = torch.Tensor([.5,.5,.5]).view(1,3,1,1).to(self.dtype).to(self.device)
        self.image_std = torch.Tensor([.5,.5,.5]).view(1,3,1,1).to(self.dtype).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
    def preprocess(self, images: image_type_mixin)->torch.Tensor:
        return base_preprocess(images, self.image_mean, self.image_std, self.device, self.dtype,im_size=self.im_size)
    def forward(self, images: image_type_mixin)->torch.Tensor:
        """
        encode the image
        """
        image_tensors = self.preprocess(images)
        output = self.model(image_tensors, output_hidden_states=True)
        features = output.hidden_states[self.layer]
        features = self.norm(features)
        return features
    def encode(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
class CLIP_Text_Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device, self.dtype = device_and_dtype(kwargs)
        clip, _ = load_clip("ViT-L/14", device = self.device)
        self.clip = clip.to(self.dtype)
        self.preprocess = CLIPProcessor.from_pretrained("/home/zhengboyang/.cache/clip/")
        self.clip.eval()
        self.im_size = kwargs.get('im_size', 224)
        for param in self.clip.parameters():
            param.requires_grad = False
    def forward(self, texts:Union[str,list[str]])->torch.Tensor:
        """
        encode the image
        """
        tensor_text = self.preprocess(texts, return_tensors="pt", padding = 'max_length').input_ids.to(self.device)
        features = self.clip.encode_text(tensor_text) # shape: [1, 768]
        features = features.unsqueeze(0) # add batch dimension
        return features
    def encode(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
class CLIP_Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device, self.dtype = device_and_dtype(kwargs)
        self.feature_type = kwargs['feature_type']
        assert self.feature_type in ['global', 'local'], "feature_type must be global or local, but got {}".format(self.feature_type)
        clip, self.clip_preprocess = load_clip("ViT-L/14", device = self.device)
        self.clip = clip.visual # the visual part of the clip model
        self.clip = self.clip.to(self.dtype)
        self.clip.eval()
        self.im_size = kwargs.get('im_size', 224)
        self.mean=torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1).to(self.dtype).to(self.device)
        self.std= torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(self.dtype).to(self.device)
        for param in self.clip.parameters():
            param.requires_grad = False
    def preprocess(self, images: image_type_mixin)->torch.Tensor:
        return base_preprocess(images, self.mean, self.std, self.device, self.dtype, self.im_size)
    def forward(self, images: image_type_mixin)->torch.Tensor:
        """
        encode the image
        """
        image_tensors = self.preprocess(images)
        if self.feature_type == 'global': # global feature (1 x dim)
            features = self.clip(image_tensors)
        elif self.feature_type == 'local': # all local features (num_patches x dim)
            features = self.clip.forward_patch_features(image_tensors)
        return features
    def encode(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
class MAE_Encoder(nn.Module):
    """
    using the encoder of Masked AutoEncoder
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device, self.dtype = device_and_dtype(kwargs)
        self.model = ViTMAEModel.from_pretrained(kwargs['ckpt_path'])
        self.noise = torch.arange(196).to(self.device) #default setting
        #self.noise = torch.randn_like(self.noise.float()).to(self.device) #default setting
        self.model = self.model.to(self.dtype).to(self.device)
        self.model.eval()
        mask_ratio = kwargs.get('mask_ratio',0.0)
        self.model.config.mask_ratio = mask_ratio
        self.processor = ViTImageProcessor.from_pretrained(kwargs['ckpt_path'])
        image_mean =  [
        0.485,
        0.456,
        0.406
        ]
        self.image_mean = torch.tensor(image_mean).view(1,3,1,1).to(self.dtype).to(self.device)
        image_std = [
            0.229,
            0.224,
            0.225
        ]
        self.im_size = kwargs.get('im_size', 224)
        self.image_std = torch.tensor(image_std).view(1,3,1,1).to(self.dtype).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
    def preprocess(self, images: image_type_mixin)->torch.Tensor:
        return base_preprocess(images, self.image_mean, self.image_std, self.device, self.dtype)
    def forward(self, images: image_type_mixin)->torch.Tensor:
        """
        encode the image
        """
        if images is None or len(images) == 0:
            return []
        image_tensors = self.preprocess(images)
        noise = self.noise.unsqueeze(0).expand(image_tensors.shape[0],-1)
        output = self.model(image_tensors,noise=noise)
        features = output.last_hidden_state
        #print(output.ids_restore, features.shape)
        return features
class VQ_Encoder(nn.Module):
    """
    use VQGAN model to encode the image
    
    This VQ_Encoder support vqgan from ldm
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if 'config_path' in kwargs:
            config_path = kwargs['config_path']
            config = OmegaConf.load(config_path)
        else:
            config = kwargs['config']
        vqgan = instantiate_from_config(config)
        if isinstance(vqgan, VQModel):
            vqgan.init_from_ckpt(kwargs['ckpt_path'])
            if kwargs.get('reshape',False):
                #if reshape is True, the output latent will be h x w, otherwise it will be h*w in raster scan order
                vqgan.quantize.sane_index_shape = kwargs['reshape']
        else:
            raise NotImplementedError(
                "VQGAN model type {} not supported".format(type(vqgan))
            )
        vqgan.eval()
        self.device, self.dtype = device_and_dtype(kwargs)
        self.vqgan = vqgan.to(self.dtype).to(self.device)
        # if quantize, we return the code instead of latent
        self.quantize = kwargs['quantize'] if 'quantize' in kwargs else False
        self.im_size = kwargs.get('im_size', 224)
        self.image_mean = torch.Tensor([.5,.5,.5]).view(1,3,1,1).to(self.dtype).to(self.device)
        self.image_std = torch.Tensor([.5,.5,.5]).view(1,3,1,1).to(self.dtype).to(self.device)
        #freeze the model
        for param in self.vqgan.parameters():
            param.requires_grad = False
    def preprocess(self, images: image_type_mixin)->torch.Tensor:
        return base_preprocess(images, self.image_mean, self.image_std, self.device, self.dtype,self.im_size)
    def forward(self, images: image_type_mixin)->torch.Tensor:
        """
        encode the image
        """
        if images is None or images is []:
            return None
        image_tensors = self.preprocess(images)
        latents = self.vqgan.encoder(image_tensors)
        
        if isinstance(self.vqgan, VQModel):
            latents = self.vqgan.quant_conv(latents)
        if self.quantize:
            codes, _, log = self.vqgan.quantize(latents)
            codes = log['min_encoding_indices'] if isinstance(log,dict) else log[-1]
            codes = codes.view(codes.shape[0],-1)
            return codes
        else:
            # latent now are in shape [bsz, latent_dim, h, w]
            latents = latents.view(latents.shape[0], latents.shape[1], -1) # bsz x latent_dim x (h*w)
            latents = latents.permute(0,2,1) # bsz x (h*w) x latent_dim
            return latents
    def encode(self, *args, **kwargs)->torch.Tensor:
        return self.forward(*args, **kwargs)
class DinoV2_Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device, self.dtype = device_and_dtype(kwargs)
        ckpt = kwargs['ckpt_path']
        # currently only support default torch.hub.load
        dino_model = Dinov2Model.from_pretrained(ckpt)
        self.dino_model = dino_model.to(self.dtype).to(self.device) # dino model does not support bf16 
        self.dino_model.eval()
        self.preprocesser = AutoImageProcessor.from_pretrained(ckpt)
        image_mean =  [
          0.485,
          0.456,
          0.406
        ]
        image_std =  [
          0.229,
          0.224,
          0.225
        ]
        self.image_mean = torch.tensor(image_mean).view(1,3,1,1).to(self.dtype).to(self.device)
        self.image_std = torch.tensor(image_std).view(1,3,1,1).to(self.dtype).to(self.device)
        self.clear_cls = kwargs.get('clear_cls', False)
        self.max_num_tokens = 256 if self.clear_cls else 257
        self.vision_tokens = self.max_num_tokens 
        self.im_size = kwargs.get('im_size', 224)
        for param in self.dino_model.parameters():
            param.requires_grad = False
    def preprocess(self, images: image_type_mixin)->torch.Tensor:
        return base_preprocess(images, self.image_mean, self.image_std, self.device, self.dtype, 224)

    def forward(self, images: image_type_mixin)->torch.Tensor:
        """
        encode the image
        """
        if (images is None) or (isinstance(images,list) and len(images) == 0):
            return []
        #if not self.dino_model.dtype == torch.float32:
        #    self.dino_model = self.dino_model.to(torch.float32) # only support float32
        image_tensors = self.preprocess(images)
        features = self.dino_model(image_tensors).last_hidden_state.to(self.dtype)
        if self.clear_cls:
            features = features[:,1:] # discard the [cls] token
        return features
class Mocov3_Encoder(nn.Module):
    """
    discarded, you can use that with proper setup from https://github.com/facebookresearch/moco-v3
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device, self.dtype = device_and_dtype(kwargs)
        model_name = kwargs['model_name']
        moco_model = MoCo_ViT(
            partial(vits.__dict__[model_name], stop_grad_conv1=True),
            256, 4096, .2) # using moco_vit base model's default setting
        ckpt_path = kwargs['ckpt_path']
        print('[!] loading moco model from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        for k in list(state_dict.keys()):
            new_k = k.replace('module.', '')
            state_dict[new_k] = state_dict[k]
            state_dict.pop(k)
        #print(state_dict.keys())
        moco_model.load_state_dict(state_dict, strict=True)
        self.moco_model = moco_model.base_encoder.to(self.dtype).to(self.device)
        self.moco_model.head = torch.nn.Identity()
        self.moco_model.global_pool = None
        self.moco_model.eval()
        print('[!] moco model loaded',self.moco_model.global_pool)
        self.transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        for param in self.moco_model.parameters():
            param.requires_grad = False
    def preprocess(self, images: image_type_mixin)->torch.Tensor:
        pil_images = load_and_transform_image_data(images)
        pil_images = [image.convert('RGB').resize((224,224), Image.BICUBIC) for image in pil_images]
        image_tensors = torch.stack(
            [self.transform(image) for image in pil_images]
        ).to(self.dtype).to(self.device)
        return image_tensors
    def forward(self, images: image_type_mixin)->torch.Tensor:
        """
        encode the image
        """
        image_tensors = self.preprocess(images)
        features = self.moco_model(image_tensors)
        return features


"""-------------------------- Adapter Modules --------------------------"""
"""We only use Linear_Adapter in our final experiments, but you can choose freely from the following adapters"""
class Linear_Adapter(nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        device, dtype = device_and_dtype(kwargs)
        kwargs['device'] = device
        kwargs['dtype'] = dtype
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.in_features
    def forward(self, x, *args, **kwargs):
        return super().forward(x)
class LoRA_Linear_Adapter(nn.Module):
    """
    This adapter is used to apply Low-Rank adaptation to the linear layer. Now the linear layer is the multiplication of two matrices of shape in_features x hidden_dim and hidden_dim x out_features. (hidden_dim < in_features, out_features)
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device, self.dtype = device_and_dtype(kwargs)
        self.hidden_dim = kwargs['hidden_dim']
        self.in_features = kwargs['in_features']
        self.out_features = kwargs['out_features']
        self.model = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_features),
        )
        self.model = self.model.to(self.dtype).to(self.device)
    def forward(self, x, *args, **kwargs):
        return self.model(x)
class Embedding_Adapter(nn.Embedding):
    """
    Receive discrete int code and output the embedding
    """
    def __init__(self, *args, **kwargs) -> None:
        device, dtype = device_and_dtype(kwargs)
        kwargs['device'] = device
        kwargs['dtype'] = dtype
        super().__init__(*args, **kwargs)
        self.hidden_dim = 1 # Embedding: [N] \to R^d
    def forward(self, x, *args, **kwargs):
        return super().forward(x)
class Identity_Adapter(nn.Module):
    """
    simply do nothing
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        device, dtype = device_and_dtype(kwargs)
        kwargs['device'] = device
        kwargs['dtype'] = dtype
        self.dtype = dtype
        self.device = device
        self.hidden_dim = -1 # Identity: R^d \to R^d, d is arbitrary
    def forward(self, x, *args, **kwargs):
        return x
class PInv_Adapter(nn.Module):
    """
    This adapter is a linear adapter, but with weights fixed to another linear layer's pseudo-inverse provided in the forward process.
    Given a linear adapter $x \\to Ax + b$, the pseudo-inverse adapter performs $Ax + b \\to x$ (if A is invertible, otherwise it performs the pseudo-inverse of A)
    
    Note that this pseudo-inverse operation keeps gradients, so gradient passing throught this PInv_adapter will affects the linear adapter.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        device, dtype = device_and_dtype(kwargs)
        kwargs['device'] = device
        kwargs['dtype'] = dtype
        self.dtype = dtype
        self.device = device
        self.hidden_dim = -1 # Pseudo Inverse: R^m \to R^d
        self.placeholder_param = nn.Parameter(torch.zeros(1,1))
    def forward(self, x, *args, **kwargs):
        adapter = kwargs['adapter']
        adapter_weight = adapter.weight
        adapter_bias = adapter.bias
        Pinv_adapter_weight = torch.pinverse(adapter_weight.float()).to(self.dtype).to(self.device)
        x -= adapter_bias # x is of shape batch_size \times seq_len \times m
        x = x @ Pinv_adapter_weight.T
        return x # x \in R^m, Pinv_adapter \in R^{m \times d}
class Truncate_Adapter(nn.Module):
    """
    This adapter is used to truncate the input tensor from bsz x s1 x hidden_dim to bsz x s2 x hidden_dim
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.truncate = kwargs['truncate']
        assert self.truncate in ['local', 'global'], "truncate must be local or global, but got {}".format(self.truncate)
        self.truncate_ratio = kwargs['truncate_ratio']
        self.hidden_dim = -1
        in_features = kwargs['in_features']
        out_features = kwargs['out_features']
        self.model = nn.Linear(in_features, out_features)
        self.device, self.dtype = device_and_dtype(kwargs)
        self.model = self.model.to(self.dtype).to(self.device)
    def forward(self, x, *args, **kwargs):
        seq_len = x.shape[1]
        if self.truncate == 'local':
            feature = x[:,:int(seq_len * (1 - self.truncate_ratio))]
        else:
            if self.truncate_ratio >= .5:
                interval = int(1/(1-self.truncate_ratio))
                feature = x[:,::interval]
            else:
                interval = int(1/self.truncate_ratio)
                index_x = (torch.arange(0,seq_len).to(self.device) % interval) != 0
                feature = x[:,index_x]
        return self.model(feature)

"""-------------------------- Decoder Modules --------------------------"""
class MAE_Decoder(nn.Module):
    """
    using the decoder of the Masked AutoEncoder
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device, self.dtype = device_and_dtype(kwargs)
        model = ViTMAEForPreTraining.from_pretrained(kwargs['ckpt_path'])
        self.decoder = model.decoder
        self.unpatchify = model.unpatchify
        self.unprocess = T.Compose([
            T.Normalize(mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225],std=[1/0.229,1/0.224,1/0.225]),

        ])
        self.decoder = self.decoder.to(self.dtype).to(self.device)
        self.default_idsrestore = torch.arange(196).to(self.device).unsqueeze(0)
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
    def forward(self, latent, ids_restore = None)->torch.Tensor:
        """
        encode the image
        """
        bsz = latent.shape[0]
        if ids_restore is None:
            ids_restore = self.default_idsrestore.expand(bsz,-1)
        features = self.decoder(latent,ids_restore=ids_restore).logits
        pixels = self.unpatchify(features)
        pixels = self.unprocess(pixels)
        return pixels
class VQ_Decoder(nn.Module):
    """
    use VQGAN model to encode the image
    
    This VQ_Encoder support vqgan from ldm
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if 'config_path' in kwargs:
            config_path = kwargs['config_path']
            config = OmegaConf.load(config_path)
        else:
            config = kwargs['config']
        vqgan = instantiate_from_config(config)
        if isinstance(vqgan, VQModel):
            vqgan.init_from_ckpt(kwargs['ckpt_path'])
            if kwargs.get('reshape',False):
                vqgan.quantize.sane_index_shape = kwargs['reshape']
        else:
            raise NotImplementedError(
                "VQGAN model type {} not supported".format(type(vqgan))
            )
        vqgan.eval()
        self.device, self.dtype = device_and_dtype(kwargs)
        self.vqgan = vqgan.to(self.dtype).to(self.device)
        self.quantize = kwargs['quantize'] if 'quantize' in kwargs else False
        self.im_size = kwargs['im_size'] if 'im_size' in kwargs else 256
        self.reshape_size = list(kwargs.get('reshape',None))
        self.unprocess = T.Compose([
            T.Lambda(lambda x: x * 0.5 + 0.5),
        ])
        #freeze the model
        for param in self.vqgan.parameters():
            param.requires_grad = False

    def forward(self, latents: torch.Tensor)->torch.Tensor:
        """
        we assume the the latents are quantized
        """
        self.vqgan: VQModel
        if self.reshape_size is not None:
            # latents: bsz x (h*w) x latent_dim
            if latents.dtype == torch.long or latents.dtype == torch.int8:
                latents = latents.view(latents.shape[0],*self.reshape_size) # bsz x h x w
            else:
                latents = latents.permute(0,2,1) # bsz x latent_dim x (h*w)
                latents = latents.reshape(latents.shape[0],latents.shape[1],*self.reshape_size) # bsz x latent_dim x h x w
        if latents.dtype == torch.long or latents.dtype == torch.int8: # quantized
            features = self.vqgan.decode_code(latents)
        else:
            features = self.vqgan.decode(latents)
        pixels = self.unprocess(features)
        return pixels
    def encode(self, *args, **kwargs)->torch.Tensor:
        return self.forward(*args, **kwargs)
class BEiT_Decoder(nn.Module):
    """
    Uses the decoder of dalle model and a linear head provided by BEiT.
    download dalle model from https://cdn.openai.com/dall-e/decoder.pkl
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.device, self.dtype = device_and_dtype(kwargs)
        ckpt_path = kwargs['ckpt_path']
        decoder_path = kwargs['dalle_decoder_path']
        model = BeitForMaskedImageModeling.from_pretrained(ckpt_path)
        self.norm = nn.Identity() # do nothing
        self.head = model.lm_head
        self.decoder:Decoder = load_model(decoder_path) # load the dalle decoder
        # make every param tensor in decoder to be contiguous
        for param in self.decoder.parameters():
            param.data = param.data.contiguous()
        self.model = nn.Sequential(self.norm, self.head)
        self.model = self.model.to(self.dtype).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.decoder = self.decoder.to(torch.float32).to(self.device) # dall_e decoder only support float32
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
    def forward(self, latent:torch.Tensor)->torch.Tensor:
        """
        encode the image
        """
        latent = latent[:,1:] # discard the [cls] token
        logits = self.model(latent)
        pred = logits.argmax(-1)
        pred = pred.reshape(-1, 14, 14).contiguous() # from seq to image
        """
        copied from https://github.com/openai/DALL-E/blob/master/notebooks/usage.ipynb
        """
        z = F.one_hot(pred, num_classes=8192).permute(0, 3, 1, 2).to(self.device).to(self.dtype)
        x_stats = self.decoder(z)
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec
    def encode(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
def Init_Vision_Components(args: dict , debug_output:bool = False)->tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Given a config, initialize the vision components and place them to the right dtype and device. Also set them trainable or not according to the config.
    """
    config_path = args['config_path']
    #print(config_path)
    config = OmegaConf.load(config_path)
    encoder_config = config['encoder']
    adapter_config = config['adapter']
    encoder_config['params']['device'] = args['device']
    adapter_config['params']['device'] = args['device']
    encoder_config['params']['dtype'] = args['dtype']
    adapter_config['params']['dtype'] = args['dtype']
    encoder = instantiate_from_config(encoder_config)
    adapter = instantiate_from_config(adapter_config)
    train_encoder, train_adapter = args['train_encoder'], args['train_adapter']
    encoder.requires_grad_(train_encoder)
    adapter.requires_grad_(train_adapter)
    if config.get('deadapter',False):
        deadapter_config = config['deadapter']
        deadapter_config['params']['device'] = args['device']
        deadapter_config['params']['dtype'] = args['dtype']
        deadapter = instantiate_from_config(deadapter_config)
        deadapter.requires_grad_(train_adapter)
        train_deadapter = args.get('train_deadapter',False)
        deadapter.requires_grad_(train_deadapter)
    else:
        deadapter = None
    if config.get('vision_loss',False):
        vision_loss_config = config['vision_loss']
        vision_loss = instantiate_from_config(vision_loss_config)
    else:
        vision_loss = None
    if config.get('decoder',False):
        decoder_config = config['decoder']
        decoder_config['params']['device'] = args['device']
        decoder_config['params']['dtype'] = args['dtype']
        decoder = instantiate_from_config(decoder_config)
        train_decoder = args.get('train_decoder',False)
        decoder.requires_grad_(train_decoder)
    else:
        decoder = None
    if debug_output:
        print('🔄 encoder parameters:', count_parameters(encoder),'Need Training:',train_encoder)
        print('🔄 adapter parameters:', count_parameters(adapter),'Need Training:',train_adapter)
        print('🔄 deadapter parameters:', count_parameters(deadapter),'Need Training:',args.get('train_deadapter',False))
        print('🔄 decoder parameters:', count_parameters(decoder),'Need Training:',args.get('train_decoder',False))
    return encoder, adapter, deadapter, decoder, vision_loss