from config import *
from datasets import *
from header import *
from model import *


def parser_args():
    parser = argparse.ArgumentParser(description="train parameters for LAMM")
    parser.add_argument(
        "--cfg", type=str, default="./config/train.yaml", help="config file"
    )
    parser.add_argument(
        "--vision_config_path",type=str,default='./config/clip_config.yaml',help="config file"
    )
    # data-related configurations
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="the path that stores the data JSON",
    )
    parser.add_argument('--vision_root_path', default=None, help='image directory', action='append')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="device to use",
    )
    parser.add_argument(
        "--max_tgt_len",
        type=int,
        default=400,
        help="max length of post-image texts in LLM input",
    )
    parser.add_argument(
        "--vision_type",
        type=str,
        default="image",
        choices=("image","text"),
        help="the type of vision data",
    )
    parser.add_argument(
        "--use_system",
        default=False,
        action="store_true",
        help="whether to use system messages",
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--save_path", type=str, help="directory to save checkpoints")
    parser.add_argument("--log_path", type=str, help="directory to save logs")
    # model-related configurations
    parser.add_argument(
        "--model", type=str, default="lamm_peft", help="Model class to use"
    )
    parser.add_argument(
        "--encoder_ckpt_path",
        type=str,
        default= '/root/ZBY/ImLang/src/model_zoo/vq-f8-n256/model.ckpt',
        help="path of vision pretrained model; CLIP use default path in cache",
    )
    parser.add_argument(
        "--llm_ckpt_path",
        type=str,
        required=True,
        help="path of LLM, default: Vicuna",
    )
    parser.add_argument(
        "--delta_ckpt_path",
        type=str,
        help="path of delta parameters from previous stage; Only matter for stage 2",
    )
    # embedding configurations
    parser.add_argument("--num_vision_token", type=int, default=1, help="number of vision tokens")
    parser.add_argument("--gpt_prefix", type=str, default=None, help="prefix for GPT")
    parser.add_argument("--conv_template", type=str, default="default", help="which conversation template to use")
    parser.add_argument("--task_token_size",type=int,
                        default=10,help="size of task token")
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="number of training stage; 1 by default; 2 if delta ckpt specified",
    )
    parser.add_argument(
        "--lora_enable",
        default=False,
        action="store_true",
        help="whether to use LoRA",
    )
    parser.add_argument(
        "--train_learnable_visual_query",
        default=False,
        action="store_true",
        help="whether to train learnable visual query",
    )
    # flash attention
    parser.add_argument(
        "--use_flash_attn",
        default=False,
        action="store_true",
        help="whether to use flash attention to speed up",
    )
    # xformers
    parser.add_argument(
        "--use_xformers",
        default=False,
        action="store_true",
        help="whether to use xformers to speed up",
    )
    parser.add_argument(
        "--train_encoder",
        default=False,
        action="store_true",
        help="train encoder or not",
    )
    parser.add_argument(
        "--train_adapter",
        default=False,
        action="store_true",
        help="train adapter or not",
    )
    parser.add_argument(
        "--train_deadapter",
        default=False,
        action="store_true",
        help="train deadapter or not",
    )
    args = parser.parse_args()

    assert not (args.use_flash_attn and args.use_xformers), 'can only use one of flash attn and xformers.'
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
    #logging.getLogger().setLevel(logging.CRITICAL)
    if args['use_flash_attn']:
        from model.flash_attn_patch import replace_llama_attn_with_flash_attn
        logging.info("⚡⚡⚡ enable flash attention.")
        replace_llama_attn_with_flash_attn()

    if args['use_xformers']:
        from model.xformers_patch import replace_llama_attn_with_xformers_attn
        logging.info("xxx enable xformers attention.")
        replace_llama_attn_with_xformers_attn()

    train_data, train_iter, sampler = load_lamm_dataset(args)

    length = (
        args["epochs"]
        * len(train_data)
        // args["world_size"]
        // dschf.config["train_micro_batch_size_per_gpu"]
    )
    total_steps = args["epochs"] * len(train_data) // dschf.config["train_batch_size"]
    args["total_steps"] = total_steps
    agent = load_model(args)
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
            agent.save_model(args["save_path"], epoch_i + 1)
        for batch in train_iter:
            agent.train_model(batch, current_step=current_step, pbar=pbar)
            current_step += 1
    # save at the end of the training
    torch.distributed.barrier()
    agent.save_model(args["save_path"], 0)

    print(f"Done! Total Training time: {time.time() - start_time}")


if __name__ == "__main__":
    args = parser_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    args = vars(args)
    # arguments from command line have higher priority
    cfg.update(args)
    main(**cfg)
