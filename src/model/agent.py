from header import *
from torch.utils.tensorboard import SummaryWriter


class DeepSpeedAgent:
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args:dict = args
        self.model:nn.Module = model

        if "log_path" in args and args["log_path"]:
            self.writer = SummaryWriter(args["log_path"])
        if args["stage"] == 2:
            if not os.path.isfile(args["delta_ckpt_path"]):
                print(f'[!] delta checkpoint not found in {args["delta_ckpt_path"]}')
            else:
                self.load_stage_1_parameters(args["delta_ckpt_path"])
                print(f'[!] load stage 1 checkpoint from {args["delta_ckpt_path"]}')

        # load config parameters of deepspeed
        ds_params = args["deepspeed"]
        ds_params["scheduler"]["params"]["total_num_steps"] = self.args["total_steps"]
        ds_params["scheduler"]["params"]["warmup_num_steps"] = max(
            10, int(self.args["total_steps"] * self.args["warmup_rate"])
        )
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args),
        )

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate(batch)
        return string
    @torch.no_grad()
    def eval_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.eval()
        loss, mle_acc = self.ds_engine(batch)
        pbar.set_description(
            f"[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}"
        )
        pbar.update(1)
        if (
            self.args["local_rank"] == 0
            and self.args["log_path"]
            and current_step % self.args["logging_step"] == 0
        ):
            elapsed = pbar.format_dict["elapsed"]
            rate = pbar.format_dict["rate"]
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            if hasattr(self,'writer'):
                self.writer.add_scalar("train/loss", loss.item(), current_step)
                self.writer.add_scalar("train/token_acc", mle_acc * 100, current_step)
            logging.info(
                f"[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}"
            )

        mle_acc *= 100
        return mle_acc
    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)
        backward_loss = loss[0] if isinstance(loss, tuple) else loss
        self.ds_engine.backward(backward_loss)
        self.ds_engine.step()
        description = f"[!] loss: {round(backward_loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}"
        if isinstance(loss, tuple) and len(loss) > 1:
            for i, sub_loss in enumerate(loss[1:]):
                description += f"; loss_{i}: {round(sub_loss.item(), 4)}"
        pbar.set_description(
            description
        )
        pbar.update(1)
        if (
            self.args["local_rank"] == 0
            and self.args["log_path"]
            and current_step % self.args["logging_step"] == 0
        ):
            elapsed = pbar.format_dict["elapsed"]
            rate = pbar.format_dict["rate"]
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            if isinstance(loss, tuple):
                for i, sub_loss in enumerate(loss):
                    self.writer.add_scalar(f"train/loss_{i}", sub_loss.item(), current_step)
            else:
                self.writer.add_scalar("train/loss", loss.item(), current_step)
            #record lr
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], current_step)
            self.writer.add_scalar("train/token_acc", mle_acc * 100, current_step)
            logging.info(
                f"[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(backward_loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}"
            )

        mle_acc *= 100
        return mle_acc

    def save_model(self, path, current_step):
        # only save trainable model parameters and params in ALWAYS_SAVE_PARAMS. You can modify it in header.py
        trainable_params = [ 
            k for (k, v) in self.ds_engine.module.named_parameters() if v.requires_grad or k in ALWAYS_SAVE_PARAMS
        ]
        # get state dict on Rank 0 (NOTE: state_dict is still none in other processes)
        state_dict = None
        if self.ds_engine.zero_optimization_partition_weights():
            if self.ds_engine.zero_gather_16bit_weights_on_model_save():
                # consolidation is expensive in time and memory and therefore isn't a default
                state_dict = self.ds_engine._zero3_consolidated_16bit_state_dict()
            else:
                raise NotImplementedError
        else:
            state_dict = self.ds_engine.module.state_dict()

        # only save checkpoint in rank 0.
        if deepspeed.comm.get_rank() == 0:
            # get checkpoint
            checkpoint = OrderedDict(
                (k, state_dict[k]) for k in trainable_params
            )

            if current_step <= 0:
                torch.save(checkpoint, f"{path}/pytorch_model.pt")
            else:
                torch.save(checkpoint, f"{path}/pytorch_model_ep{current_step}.pt")
            # save tokenizer
            #self.model.llama_tokenizer.save_pretrained(path)
            # save configuration
            #self.model.llama_model.config.save_pretrained(path)
            print(f"[!] save model into {path}")
        


    def load_stage_1_parameters(self, path):
        delta_ckpt = torch.load(path, map_location=torch.device("cpu"))
        keys = self.model.load_state_dict(delta_ckpt, strict=False)
        print("[!] load stage 1 parameters from delta checkpoint with unexpected keys:", keys.unexpected_keys)
        #assert self.model.llama_proj.weight.requires_grad == True, "projection layer should be trainable"
