import os
import sys

import deepspeed.comm
#add parent directory to head of path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import deepspeed.comm
import requests
import torch
from PIL import Image, ImageFile
from header import *
from model.vision_modules import Init_Vision_Components, read_images_with_executor
from ldm.util import instantiate_from_config,get_obj_from_str
from torch.utils.data import Dataset,DataLoader
import numpy as np
from datasets.samplers import DistributedBatchSampler
from taming.modules.losses.lpips import LPIPS
from model.agent import DeepSpeedAgent
from colorama import Fore, Style, Back, init
from model.vision_modules import read_images_with_executor,load_image
from torchvision.datasets import DatasetFolder
DEFAULT_IM_SIZE = 224
init(autoreset=True)
class Imagenet_Dataset(Dataset):
    def __init__(self, data_path, read_pil:bool = False):
        super(Imagenet_Dataset, self).__init__()
        print(f"🔥loading data from {data_path}")
        self.data_path = data_path
        self.read_pil = read_pil
        if read_pil:
            deepspeed_init = deepspeed.comm.comm.is_initialized()
            files = read_images_with_executor(data_path,max_workers=16,show_tqdm=deepspeed.comm.get_rank() == 0 if deepspeed_init else True,im_size=DEFAULT_IM_SIZE,recursive=True)
            self.files = [ file[1] for file in files]
            self.names = [ file[0] for file in files]
        else:
            # read all images under subfolders
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data path not found: {data_path}")
            self.files = []
            self.names = []
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.names.append(os.path.join(root, file))
                        self.files.append(os.path.join(root, file))
        print(f"🔥 Loaded {len(self.files)} images")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        return self.files[idx] , self.names[idx] 
    def collate_fn(self, batch):
        if self.read_pil:
            files = [np.ndarray(data[0]) for data in batch]
            files = np.ndarray(files).transpose(0,3,1,2)
            files = torch.Tensor(files) / 255.
        else:
            files = [load_image(data[0],return_np=True,to_tensor=True)[1] for data in batch]
        files = torch.stack(files)
        names = [data[1] for data in batch]
        return files, names
            
class SimpleWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(SimpleWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_recon = torch.abs # L1 loss
        lpips_loss = LPIPS()
        lpips_loss.eval()
        self.loss_lpips = lpips_loss # LPIPS loss
        self.beta = 1. # LPIPS weight
    def forward(self, x):
        x , _ = x # x is a tuple
        x = x.to(self.encoder.device).to(self.encoder.dtype)
        x_f = x.detach().clone()
        with torch.no_grad():
            y = self.encoder(x)
        z = self.decoder(y)
        recon_loss = self.loss_recon(z - x_f).mean()
        lpips_loss = self.loss_lpips(z, x_f).mean()
        loss = recon_loss + self.beta * lpips_loss # for MAE* mentioned in our paper
        #loss = recon_loss  # for MAE-L1
        return (loss,recon_loss,lpips_loss), 0.114
    @torch.no_grad()
    def infer(self, x):
        x, _  = x
        x = x.to(self.encoder.device).to(self.encoder.dtype)
        y = self.encoder(x)
        z = self.decoder(y)
        return z
def get_data(args):
    data_path  = args["data_path"]
    #file_path = os.path.join(data_path, "images.npy")
    #dict_path = os.path.join(data_path, "info_dict.npy")
    dataset = Imagenet_Dataset(data_path)
    sampler = torch.utils.data.RandomSampler(dataset)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = (
        args["world_size"] * args["dschf"].config["train_micro_batch_size_per_gpu"]
    )
    print(f"Batch size: {batch_size}, world size: {world_size},rank: {rank}")
    batch_sampler = DistributedBatchSampler(sampler, batch_size, True, rank, world_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=32,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
    return dataset, dataloader, sampler

def build_model(vision_config_path,device='cuda'):
    args = {
        'config_path': vision_config_path,
        'train_encoder': False,
        'train_adapter': False,
        'device': device,
        'dtype': 'torch.bfloat16',
    }
    encoder, _, _, decoder, _ = Init_Vision_Components(args)
    encoder.eval()
    decoder.train()
    encoder.requires_grad_(False)
    decoder.requires_grad_(True)
    return encoder, decoder
def save_model(agent:DeepSpeedAgent ,path:str, epoch:int):
    state_dict = agent.ds_engine.module.state_dict()
    checkpoint = OrderedDict(
                (k.replace("decoder.", "", 1)
                , state_dict[k]) for k in state_dict if k.startswith("decoder") # only save decoder
            )
    #print('saving checkpoint:',checkpoint.keys())
    if epoch <= 0:
        torch.save(checkpoint, f"{path}/pytorch_model.pt")
    else:
        torch.save(checkpoint, f"{path}/pytorch_model_ep{epoch}.pt")
    # save tokenizer
    #self.model.llama_tokenizer.save_pretrained(path)
    # save configuration
    #self.model.llama_model.config.save_pretrained(path)
    print(f"[!] save model into {path}")

def load_model(decoder, decoder_path:str):
    decoder.load_state_dict(torch.load(decoder_path),strict=True)
    return decoder


def parser_args():
    parser = argparse.ArgumentParser(description="train parameters for LAMM")
    parser.add_argument(
        "--cfg", type=str, default="./config/train.yaml", help="config file"
    )
    parser.add_argument(
        "--vision_config",type=str, default="./config/vision.yaml", help="vision config file"
    )
    # data-related configurations
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="the path that stores the data JSON",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or eval",
        choices=["train", "eval"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to run the model",
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--save_path", type=str, help="directory to save checkpoints")
    parser.add_argument("--log_path", type=str, help="directory to save logs")
    parser.add_argument("--delta_ckpt_path", type=str, help="path to delta checkpoint")
    # model-related configurations
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="number of training stage; 1 by default; 2 if delta ckpt specified",
    )

    args = parser.parse_args()

    if args.local_rank == 0:
        print(
            "Arguments: \n{}".format(
                json.dumps(vars(parser.parse_args()), indent=4, sort_keys=True)
            )
        )
    return args


def initialize_distributed(args):
    args["master_ip"] = os.getenv("MASTER_ADDR", "localhost")
    args["master_port"] = os.getenv("MASTER_PORT", "6000")
    args["world_size"] = int(os.getenv("WORLD_SIZE", "1"))
    args["local_rank"] = int(os.getenv("RANK", "0")) % torch.cuda.device_count()
    device = args["local_rank"] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend="nccl")


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_env(args):
    args["root_dir"] = "../"
    args["mode"] = "train"
    initialize_distributed(args)
    set_random_seed(args["seed"])


def build_directory(path):
    if os.path.exists(path):
        pass
    else:  # recursively construct directory
        os.makedirs(path, exist_ok=True)

def eval(**args):
    start_time = time.time()
    encoder,decoder = build_model(args["vision_config"],args["device"])
    #assert os.path.exists(args["delta_ckpt_path"]), f"checkpoint not found: {args['delta_ckpt_path']}"
    save_path = args["save_path"]
    if os.path.exists(args["delta_ckpt_path"]):
        decoder = load_model(decoder, args["delta_ckpt_path"])
    os.makedirs(save_path, exist_ok=True)
    encoder.eval(),decoder.eval()
    model = SimpleWrapper(encoder, decoder)
    # detect whether images.npy or images folder
    dataset = Imagenet_Dataset(args["data_path"])
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=32,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
    idx = 0
    for batch in tqdm(dataloader):
        batch_name = batch[1]
        recon_batch = model.infer(batch)
        recon_batch = recon_batch.clip(0, 1).cpu().float().numpy()
        recon_batch = (recon_batch * 255).astype(np.uint8)
        for i in range(recon_batch.shape[0]):
            img = Image.fromarray(recon_batch[i].transpose(1, 2, 0))
            save_name = batch_name[i].split("/")[-1].replace(".JPEG", ".png").replace(".jpg", ".png").replace(".jpeg", ".png")
            img.save(os.path.join(save_path, save_name))
            idx += 1
    end_time = time.time()
    # print time in 4 decimals
    print(f"🔥 Done! {idx} images reconstructed, time cost:{end_time - start_time:.4f}s")
def main(**args):
    start_time = time.time()
    config_env(args)
    build_directory(args["save_path"])
    build_directory(args["log_path"])

    # dump training settings
    with open(os.path.join(args["log_path"], "training_args.json"), "w") as fw:
        json.dump(args, fw, indent=4)

    dschf = HfDeepSpeedConfig(args["deepspeed"])
    args["dschf"] = dschf

    if args["log_path"]:
        logging.basicConfig(
            format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
            level=logging.INFO,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode="w",
        )

    train_data, train_iter, sampler = get_data(args)

    length = (
        args["epochs"]
        * len(train_data)
        // args["world_size"]
        // dschf.config["train_micro_batch_size_per_gpu"]
    )
    total_steps = args["epochs"] * len(train_data) // dschf.config["train_batch_size"]
    args["total_steps"] = total_steps
    encoder, decoder = build_model(args["vision_config"], args["device"])
    model = SimpleWrapper(encoder, decoder)
    agent = DeepSpeedAgent(model, args)
    torch.distributed.barrier()
    with open(os.path.join(args["log_path"], "training_args.yaml"), "w") as fw:
        yaml.dump(args, fw)
    # begin to train
    pbar = tqdm(total=length,disable= (args["local_rank"] != 0))  # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args["epochs"])):
        if (
            epoch_i % max(args["epochs"] // 5, 1) == 0
        ):  # save epoch0 & save 5 models at most
            if deepspeed.comm.get_rank() == 0:
                save_model(agent, args["save_path"], epoch_i + 1)
        for batch in train_iter:
            agent.train_model(batch, current_step=current_step, pbar=pbar)
            current_step += 1
    # save at the end of the training
    torch.distributed.barrier()
    if deepspeed.comm.get_rank() == 0:
        save_model(agent, args["save_path"], 0)

    print(f"Done! Total Training time: {time.time() - start_time}")


if __name__ == "__main__":
    args = parser_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    args = vars(args)
    # arguments from command line have higher priority
    cfg.update(args)
    if cfg["mode"] == "train":
        main(**cfg)
    elif cfg["mode"] == "eval":
        eval(**cfg)