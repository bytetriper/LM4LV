import deepspeed.comm
import torch
from PIL import Image, ImageFile
from torch.nn.utils import rnn
from peft.peft_model import PeftModel, PeftModelForCausalLM
import conversations
from header import *
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LlamaTokenizer,AutoTokenizer
from .modeling_llama import LlamaForCausalLM,get_valid_mask
from model.vision_modules import Init_Vision_Components, count_parameters,load_and_transform_data_mixin,load_image,read_images_with_executor
from colorama import Fore, Back, Style, init
from model.losses import *
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_COLOR = Back.WHITE + Fore.BLUE + Style.BRIGHT
init(autoreset=True)
VISION_TAGS = {
    "pos": {
        "image": "<image>",
        "text": "<text>"
    },
    "sov": {
        "image": "<Img>",
        "text": "<Txt>"
    },
    "eov": {
        "image": "</Img>",
        "text": "</Txt>"
    },
    "query": {
        "image": "<query>",
        "text": "<query>"
    },
    "task":{
        "image": "<task>",
        "text": "<task>"
    }
}
class LM4LVStoppingCriteria(StoppingCriteria):

    def __init__(self, stops, input_ids):
        """intialize stopping criteria

        :param list stops: list of stop tokens
        :param list input_ids: input ids
        """
        super().__init__()
        self.stops = [torch.tensor(stop).to('cuda') for stop in stops]
        self.bsz = input_ids.shape[0]
        self.stop_flag = [0] * self.bsz
    def check_stop(self, input_ids):
        """check whether to stop generation

        :param list input_ids: input token ids
        :return bool: stop or not
        """
        for stop in self.stops:
            if len(stop) > len(input_ids):
                continue
            if torch.all((stop == input_ids[-len(stop):])).item():
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        """call function of stop creteria

        :param torch.LongTensor output_ids: output token ids
        :return bool: stop or not
        """
        self.stop_flag = [0] * self.bsz
        flag = 1
        for id, output_id in enumerate(output_ids):
            if self.stop_flag[id] == 1:
                continue
            if self.check_stop(output_id):
                self.stop_flag[id] = 1
            else:
                flag = 0
        if flag == 1:
            return True
        return False
    
def find_all_substring(string: str, substring: str) -> list[int]:
    """
    Find all substring in string
    :param string: string
    :param substring: substring
    :return: list of start index of substring
    """
    start = 0
    index_list = []
    while True:
        start = string.find(substring, start)
        if start == -1:
            break
        index_list.append(start)
        start += 1
    return index_list


def build_one_instance(
    tokenizer,
    conversation,
    vision_type: str,
    #vision_instances: list = [],
    num_vision_tokens: int = 1,
    num_task_tokens: int = 1,
    template=conversations.default_conversation,
    gpt_prefix: str = None,
    generation_mode:bool = False
):
    """build one instance for training; text part

    :param class tokenizer: text tokenizer
    :param list conversation: list of conversation
    :param str vision_type: type of vision data, defaults to 'image'
    :raises Exception: Exception if wrong role included
    :return list: conversation text list, input token ids, target token ids
    """
    
    pos = VISION_TAGS["pos"][vision_type]
    eov = VISION_TAGS["eov"][vision_type]
    sov = VISION_TAGS["sov"][vision_type]
    query = VISION_TAGS["query"][vision_type]
    task = VISION_TAGS["task"][vision_type]
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    vision_placeholder = (num_vision_tokens - 1) * [VISION_CONTROL_ID]
    task_placeholder = (num_task_tokens - 1) * [VISION_CONTROL_ID]
    im_token_id = tokenizer.convert_tokens_to_ids(pos) # TODO
    query_token_id = tokenizer.convert_tokens_to_ids(query)
    task_token_id = tokenizer.convert_tokens_to_ids(task)
    #print("im_token_id", im_token_id, "query_token_id", query_token_id, "task_token_id", task_token_id)
    for i in range(turn_num):
        turn = conversation[i]
        role = turn["from"]
        # replace <pos> with corresponding image codes
        processed_text = turn["value"]
        #subsitute all <image> with <img><image></img>
        processed_text = processed_text.replace(pos, sov + pos + eov)
        processed_text = processed_text.replace(query, sov + query + eov) # also substitute <query> with <img><query></img>
        turn["value"] = processed_text
        #print("processed_text", processed_text)
        assert (i != 0) or (
            role in ['human','none']), "First turn must be human or none, not {}".format(role)
        if role == "human":
            # text = "{}: ".format(template.roles[0]) + turn["value"] + "\n### {}:".format(template.roles[1])
            if i!=0:
                text = "{}: {}\n{} {}: ".format(template.roles[0], turn["value"],
                                            template.sep, template.roles[1])
            else:
                text = "{}\n{} {}: ".format(turn["value"],
                                            template.sep, template.roles[1])
            if gpt_prefix is not None:
                text = text + gpt_prefix
            #print("HUMAN:", text)
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
        elif role == "gpt":
            if i == turn_num - 1 and generation_mode:
                #print("Generation mode, no need to add sep in the last gpt turn")
                text = turn["value"]
            else:
                text = turn["value"] + "\n{}".format(template.sep2 if (
                    template.sep2 is not None) else template.sep)
            #print("GPT:", text)
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
        elif role == 'none':
            #print("None role", turn["value"],i ,turn_num ,generation_mode)
            if i == turn_num - 1 and generation_mode:
                #print("Generation mode, no need to add sep in the last gpt turn")
                text = turn["value"]
            else:
                text = turn["value"] + "\n"
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
        else:
            raise Exception(f"'{role}' is a Wrong Role!!!")
        # substitute all im_token_id with corresponding image codes
        j = 0
        while j < len(one_input_id):
            if one_input_id[j] == im_token_id:
                #print("find im_token_id at", j,'jumps to', j+num_vision_tokens-1)
                one_input_id = one_input_id[:j] + [VISION_START_ID] + vision_placeholder + one_input_id[j+1:]
                j = j + num_vision_tokens - 1
            elif one_input_id[j] == query_token_id:
                #print("find query_token_id at", j,'jumps to', j+num_vision_tokens-1)
                one_input_id = one_input_id[:j] + [QUERY_START_ID] + vision_placeholder + one_input_id[j+1:]
                j = j + num_vision_tokens - 1
            elif one_input_id[j] == task_token_id:
                #print("find task_token_id at", j,'jumps to', j+num_task_tokens-1)
                one_input_id = one_input_id[:j] + [TASK_START_ID] + task_placeholder + one_input_id[j+1:]
                j = j + num_task_tokens - 1
            #print(one_input_id[j], end=' ')
            j += 1
        #print('input_ids', one_input_id)
        input_ids += one_input_id
        if role == "human":
            target_ids += [INVALID_TOKEN_ID] * len(one_input_id) # do not perform loss regression on human prompt
        else:
            # to avoid calculating loss on vision tokens while distingushing human prompt and vision tokens
            one_target_id = [VISION_CONTROL_ID if id < 0 else id for id in one_input_id]
            target_ids += one_target_id
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    #print(text_list)
    #print('input_ids', input_ids)
    #print('target_ids', target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer,
                            batch_of_conversations,
                            max_tgt_len,
                            vision_type="image",
                            num_vision_tokens: int = 1,
                            num_task_tokens: int = 1,
                            #vision_instances: list[list[str]] = [],
                            template=conversations.default_conversation,
                            gpt_prefix: str = None,
                            generation_mode:bool = False
                            ):
    """build one batch of instance for training

    :param class tokenizer: text tokenizer
    :param list batch_of_conversations: batch of conversations
    :param int max_tgt_len: max token length of after vision tokens
    :param str vision_type: type of vision data, defaults to 'image'
    :return list: input token ids, target token ids, attention mask
    """
    batch_input_ids, batch_target_ids = [], []
    for i, conversation in enumerate(batch_of_conversations):
        text, one_input_ids, one_target_ids = build_one_instance(
            tokenizer,
            conversation,
            vision_type=vision_type,
            num_vision_tokens=num_vision_tokens,
            num_task_tokens=num_task_tokens,
            template=template,
            gpt_prefix=gpt_prefix,
            generation_mode=generation_mode
        )
        #print("text", text)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids,
                                batch_first=True,
                                padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids,
                                batch_first=True,
                                padding_value=INVALID_TOKEN_ID)

    assert input_ids.size() == target_ids.size()
    if max_tgt_len <= len(input_ids[0]):
        print(Fore.RED + "Warning: max_tgt_len is smaller than the length of input_ids[0], which is {}".format())
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    # for any input_id < 0, we set the corresponding attention mask to be 0
    #attention_mask = attention_mask & (input_ids >= 0)
    #print("input_ids: ", input_ids.shape, "target_ids: ", target_ids.shape,
    #      "attention_mask: ", attention_mask.shape)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def make_prompt_start(use_system=False,
                    task_type="normal",
                    template=conversations.default_conversation):
    """make starting prompt

    :param bool use_system: whether to use system message, defaults to False
    :param str task_type: task type of current sample, defaults to 'normal'
    :return str: resulting starting prompt
    """
    PROMPT_START = f'{template.sep} {template.roles[0]}: '
    if use_system:
        if task_type == "normal":
            return f"{template.system}\n\n" + PROMPT_START
        else:
            if template.sys_temp is None:
                return [
                    f"{conversations.conversation_dict[task]}\n\n" +
                    PROMPT_START for task in task_type
                ]
            else:
                return [
                    template.sys_temp.format(
                        system_message=conversations.conversation_dict[task]) +
                    PROMPT_START for task in task_type
                ]
    else:
        return PROMPT_START


class LM4LVPEFTModel(nn.Module):
    """A general PEFT class for LM4LV"""

    def __init__(self, **args):
        super(LM4LVPEFTModel, self).__init__()
        # check if deepspeed is initialized
        if not deepspeed.comm.comm.is_initialized():
            self.cprint = print # if deepspeed is not initialized, we print directly
            self.cprint("⚠️⚠️⚠️ DeepSpeed is not initialized. ⚠️⚠️⚠️")
        self.args = args
        self.device = args.get('device',torch.cuda.current_device())
        
        llm_ckpt_path = args["llm_ckpt_path"]
        use_system = args.get('use_system', False)
        self.use_system = use_system
        self.conv_template = conversations.conv_templates[args[
            'conv_template']] if 'conv_template' in args else conversations.default_conversation
        device = args["device"]
        self.rank = args["rank"] if "rank" in args else 0
        
        
        ## training settings
        self.log_path = args["log_path"] if "log_path" in args else './logs'
        self.cprint(f"Logging to {self.log_path}")
        self.vision_type = args[
            "vision_type"] if "vision_type" in args else "image"
        vision_config_path = args['vision_config_path']
        assert os.path.isfile(vision_config_path), "vision_config_path: {} is not a file".format(vision_config_path)
        self.cprint(
            f"🔥 Initializing vision modules from {vision_config_path} ..."
        )
        self.num_vision_token = args["num_vision_token"]
        vision_args = {
            "config_path": vision_config_path,
            "device": device,
            "dtype": 'torch.bfloat16', 
            # we manually set the vision encoder dtype to be bfloat16, you can change it to torch.float32 or whatever you like, this will be further overrided by the dtype in the deepspeed config
            "train_encoder": args["train_encoder"],
            "train_adapter": args["train_adapter"],
            "train_deadapter": args["train_deadapter"],
        }
        # initialize vision components
        self.visual_encoder, self.adapter,self.deadapter, self.decoder, self.vision_loss = Init_Vision_Components(
            vision_args ,
            debug_output = deepspeed.comm.get_rank() == 0
        )
        # if you self-defined a vision encoder, you should make sure it contains a im_size attribute
        self.im_size = self.visual_encoder.im_size 
        # if you self-defined a vision encoder, you should make sure it contains a hidden_dim attribute, even if it's meaningless(like -1).
        self.visual_hidden_dim = self.adapter.hidden_dim
        encoder_params, adapter_params = count_parameters(self.visual_encoder), count_parameters(self.adapter)
        self.cprint("📢 Using conversation template: ", self.conv_template)
        self.cprint("📢 Enabling system message: ", use_system,
            self.conv_template.system if self.use_system else "None")
        self.gpt_prefix = args.get('gpt_prefix', None)
        self.cprint("📢 Using GPT prefix: ", self.gpt_prefix)
        self.cprint(
            f"🔄 Vision encoder initialized with {encoder_params} parameters, adapter initialized with {adapter_params} parameters"
        )
        self.cprint(f"Initializing language decoder from {llm_ckpt_path} ...")
        # add the lora module, though we do not need it, we still support a LoRA finetuning setup.
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=self.args["lora_target_modules"],
        )
        if args.get('use_lightllm', False):
            # currently we do not support lightllm
            raise NotImplementedError
        else:
            llama_model = LlamaForCausalLM.from_pretrained(llm_ckpt_path)
            self.llama_model = get_peft_model(llama_model, peft_config)
            # temporarily disable LoRA by setting all the parameter to be untrainable
            # NOTE: you should also set alpha = 0 to assure LoRA takes no effect
            if not args.get("lora_enable", False):
                self.cprint("LORA is disabled.")
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
            self.print_trainable_parameters()
        # we also support a learnable visual query for visual generation, which is not covered in paper. Simply un-comment the following code to enbale this.
        # If enabledm we add a learnable visual query token sequence <query> to the model.
        # To use this feature, replace the desired generated token from <image> to <query> in the prompt 
        # For example, a denoising prompt would be like "Human: <image> <task> Assistant: <image>", change it to "Human: <image> <task> Assistant: <query>"
        
        # Btw, to use it in generation, you may need to write a custom generation function to handle the generation of <query> token
        """
        train_query = args.get('train_learnable_visual_query', False)
        self.learnable_visual_query = torch.nn.Parameter(
            torch.zeros(1 , self.num_vision_token ,self.llama_model.config.hidden_size, dtype=torch.bfloat16).to(self.device),
            requires_grad=train_query
        )
        self.cprint(f"Adding learnable visual query : {self.learnable_visual_query.data.shape}, Train:{train_query}")
        """
        self.task_token_size = args.get('task_token_size', 10) # default to 10
        self.learnable_task_token = torch.nn.Parameter(
            torch.zeros(1 , self.task_token_size ,self.llama_model.config.hidden_size, dtype=torch.bfloat16).to(self.device),
            requires_grad=True # always add and train task token
        )
        self.cprint(f"Adding learnable task token : {self.learnable_task_token.data.shape}")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llm_ckpt_path,
                                                            use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        self.cprint("🔄 Tokenizer initialized with vocab size: ", self.llama_tokenizer.vocab_size)
        #print info about the tokenizer
        self.cprint(f"📢 Tokenizer special tokens: {self.llama_tokenizer.special_tokens_map}, pad_token: {self.llama_tokenizer.pad_token_id}, bos_token: {self.llama_tokenizer.bos_token_id}, eos_token: {self.llama_tokenizer.eos_token_id}")
        self._init_image_tokens()
        self.cprint(
            f"🔄 Language decoder initialized with vocab size {self.llama_tokenizer.vocab_size}"
        )
        self.print_trainable_parameters()
        self.max_tgt_len = args["max_tgt_len"]
        # self.use_system = use_system
        self.use_flash_attn = args.get('use_flash_attn', False)
        self.use_xformers = args.get('use_xformers', False)
    #decorator for debugging
    def debug(func):
        def wrapper(*args, **kwargs):
            # if deepspeed backend is initialized, only rank 0 will print. Please assure deepspeed is always initialized before calling this function
            if deepspeed.comm.get_rank() == 0:
                func(*args, **kwargs)
        return wrapper
    @debug
    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"🔄 trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    @debug
    def cprint(self, *args, **kwargs):
        """
        Print only on rank 0, for debugging purposes.
        """
        print(*args, **kwargs)
    def get_vision_instances_generator(self, vision_instances: list[dict] = []) -> Generator:
        # def a generator to get vision instance
        # vision_instances: list of vision instance
        self.vision_instances = vision_instances
        def vision_instances_generator(vision_instances: list[dict] = [])-> Generator:
            for i, vision_instance in enumerate(vision_instances):
                yield i , vision_instance['embed']
        self.get_vision_instances = vision_instances_generator(vision_instances)
        return self.get_vision_instances
    def _init_image_tokens(self) -> None:
        """
        Add necessary special tokens to the tokenizer for multi-modal generation.
        """
        additional_tokens = [VISION_TAGS["pos"][self.vision_type],VISION_TAGS['query'][self.vision_type],
                            VISION_TAGS["task"][self.vision_type]]
        self.llama_tokenizer.add_tokens(
            additional_tokens)
        self.cprint (f"Add {len(additional_tokens)} image tokens {additional_tokens}")
    def embed_tokens(self, input_ids, *args, **kwargs) -> torch.Tensor:
        # for those vision_token, we insert them directly. For the text tokens, we embed them normally.
        # input_ids: bsz x s1 (x 1)
        # embeds: bsz x s1 x embed_dim
        embeds = torch.zeros(input_ids.size()[0],
                            input_ids.size()[1],
                            self.llama_model.config.hidden_size,
                            dtype=self.llama_model.dtype).to(self.device)
        target_mask = INVALID_TOKEN_ID * torch.ones(input_ids.size()[0],
                            input_ids.size()[1],dtype=torch.long).to(self.device) # init to be INVALID_TOKEN_ID
        vision_idx = (input_ids == VISION_START_ID) | (input_ids == TASK_START_ID) | (input_ids == QUERY_START_ID)
        if vision_idx.sum() > 0:
            # replace every <img>[INVALID_TOKEN_ID]*num_vision_token</img> with <img>vision_embed</img>
            sov_pos = vision_idx.nonzero()
            #print(sov_pos)
            for pos in sov_pos:
                # vis_idx : index of vision instance (int)
                # vision_embed: vision embeds/tokens
                if input_ids[pos[0], pos[1]] == QUERY_START_ID:
                    vis_idx , vision_embed = next(self.get_vision_instances) 
                    embeds[pos[0], pos[1]: pos[1]+self.num_vision_token] = self.learnable_visual_query.squeeze(0)
                    target_mask[pos[0], pos[1] - 1: pos[1] + self.num_vision_token - 1] = vis_idx # calculate vision loss on query token
                elif input_ids[pos[0], pos[1]] == VISION_START_ID:
                    vis_idx , vision_embed = next(self.get_vision_instances) 
                    if vision_embed.squeeze(0).shape[0] != self.num_vision_token:
                        if not hasattr(self, 'size_error'):
                            # this only warns you for one time
                            self.size_error = True
                            self.cprint(Fore.BLUE + Style.BRIGHT +  f"vision_embed is of shape {vision_embed.squeeze(0).shape}, but num_vision_token is {self.num_vision_token}") 
                            raise IndexError("If you're sure that the vision_embed is of correct shape, please ignore this error by commenting out this line")                        
                    embeds[pos[0], pos[1]: pos[1]+self.num_vision_token] = vision_embed.squeeze(0)[:self.num_vision_token]
                    target_mask[pos[0], pos[1] - 1: pos[1] + self.num_vision_token - 1] = vis_idx # calculate vision loss on vision token
                elif input_ids[pos[0], pos[1]] == TASK_START_ID:
                    #self.cprint('task start at', pos[0], pos[1])
                    embeds[pos[0], pos[1]: pos[1]+self.task_token_size] = self.learnable_task_token.squeeze(0) # do not calculate loss on task token
                else:
                    raise NotImplementedError('token type {} not implemented'.format(input_ids[pos[0], pos[1]]))
        token_idx = (input_ids <= self.llama_tokenizer.vocab_size + 1) & (input_ids >= 0) # text tokens
        if token_idx.sum() > 0:
            token_embeds = self.llama_model.model.embed_tokens(
                input_ids[token_idx])
            embeds[token_idx] = token_embeds
        return embeds , target_mask

    def prompt_wrap(self, input_ids, target_ids, attention_mask, use_system,
                    task_type, padding_right: bool = True):
        """
        input_ids, target_ids, attention_mask: bsz x s2
        padding_right: whether to pad right or left

        """
        PROMPT_START = task_type[0] != "simple" # if task_type is simple, we do not add any system message or chat template
        input_ids = input_ids.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2
        batch_size = input_ids.shape[0]
        if PROMPT_START:
            # return list of headers if multiple tasks
            p_before = make_prompt_start(use_system=use_system,
                                        task_type=task_type,
                                        template=self.conv_template)
            if isinstance(p_before, list):
                # this multiple task alignment is buggy, TODO: fix it
                # e.g. will lead to situation like:
                # p_before = [p1, p2]
                # p1... /p1 <pad> <pad> p_after ... /p_after <pad> ...
                # p2...      /p2  <pad> p_after ... /p_after <pad> ...
                # however, it should be :
                # p1... /p1 p_after ... /p_after <pad> <pad> ...
                # p2...     /p2 p_after ...   /p_after <pad> <pad> ...
                p_before_tokens = [
                    self.llama_tokenizer(p,
                                        return_tensors="pt",
                                        add_special_tokens=False).input_ids[0].to(
                                            self.device) for p in p_before
                ]
                # TODO: test in batch
                p_before_token_ids = rnn.pad_sequence(
                    p_before_tokens,
                    batch_first=True,
                    padding_value=self.llama_tokenizer.pad_token_id,
                )  # bsz x s1
                p_before_attn_mask = p_before_token_ids.ne(
                    self.llama_tokenizer.pad_token_id)
            else:
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(
                        self.device)  # [s1, s1...] list of batch size
                p_before_token_ids = p_before_tokens.input_ids.expand(
                    batch_size, -1)  # bsz x s1
                p_before_attn_mask = p_before_tokens.attention_mask.expand(
                    batch_size, -1)  # bsz x s1
            # peft model need deeper call
            p_before_embeds,p_before_target_mask = self.embed_tokens(
                p_before_token_ids
            )  # .expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        p_after_embeds, p_after_target_mask = self.embed_tokens(input_ids)
        #print('p_after_embeds', p_after_embeds.shape)
        bos = (torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=input_ids.device,
        ) * self.llama_tokenizer.bos_token_id)  # bsz x 1
        bos_embeds, bos_target_mask = self.embed_tokens(bos)  # bsz x 1 x embed_dim
        atts_bos = torch.ones([batch_size, 1],
                            dtype=torch.long).to(self.device)  # bsz x 1
        if PROMPT_START:
            inputs_embeds = torch.cat(
                [bos_embeds, p_before_embeds, p_after_embeds],
                dim=1)  # bsz x (1+s1+s2) x embed_dim
            # make target ids for prefix part
            empty_targets = (
                torch.ones(
                    [batch_size, 1 + p_before_embeds.size()[1]],
                    dtype=torch.long,
                ).to(self.device).fill_(INVALID_TOKEN_ID)  # 1 (bos) + s1
            )  # bsz x (1 + s1 + 1)
            final_attention_mask = torch.cat(
                [atts_bos, p_before_attn_mask, attention_mask], dim=1)
            final_target_mask = torch.cat(
                [bos_target_mask, p_before_target_mask, p_after_target_mask], dim=1)
        else:
            inputs_embeds = torch.cat(
                [bos_embeds, p_after_embeds], dim=1)
            empty_targets = (
                torch.ones(
                    [batch_size, 1],
                    dtype=torch.long,
                ).to(self.device).fill_(INVALID_TOKEN_ID)  # 1 (bos) 
            ) # bsz x (1 + s1 + 1)
            final_attention_mask = torch.cat(
                [atts_bos, attention_mask], dim=1)
            final_target_mask = torch.cat(
                [bos_target_mask, p_after_target_mask], dim=1)
        targets = torch.cat([empty_targets, target_ids],
                            dim=1)  # bsz x (1 + s1 + s2)
        assert inputs_embeds.size()[1] == targets.size()[1] == final_target_mask.size()[1] == final_attention_mask.size()[1]
        
        final_target_mask[(targets==INVALID_TOKEN_ID).roll(-1,dims=1)] = INVALID_TOKEN_ID # to avoid calculating loss on human prompt
        targets[targets==VISION_CONTROL_ID] = INVALID_TOKEN_ID # to avoid calculating loss on vision tokens
        if not padding_right:
            # call when generation
            after_embeds_lens = attention_mask.sum(dim=-1)  # bsz
            after_pad_lens = attention_mask.shape[1] - after_embeds_lens
            embeds_lens = final_attention_mask.sum(dim=-1)  # bsz
            pad_lens = final_attention_mask.shape[1] - embeds_lens
            assert (pad_lens == after_pad_lens).all(), "pad_lens: {} != after_pad_lens: {}".format(
                pad_lens, after_pad_lens) # sanity check
            inputs_embeds = torch.stack(
                [
                    torch.cat(
                    [
                        inputs_embeds[i, -pad_lens[i]:, :],
                        inputs_embeds[i, :embeds_lens[i], :],
                    ]
                ) if pad_lens[i] != 0 else inputs_embeds[i, :, :]
                for i in range(batch_size)
                ]
            )
            targets = torch.stack(
                [
                    torch.cat(
                    [
                        targets[i, -pad_lens[i]:],
                        targets[i, :embeds_lens[i]],
                    ]
                ) if pad_lens[i] != 0 else targets[i, :]
                for i in range(batch_size)
                ]
            )
            
            final_attention_mask = torch.stack(
                [
                    torch.cat(
                    [
                        final_attention_mask[i, -pad_lens[i]:],
                        final_attention_mask[i, :embeds_lens[i]],
                    ]
                ) if pad_lens[i] != 0 else final_attention_mask[i, :]
                for i in range(batch_size)
                ]
            )
            
            final_target_mask = torch.stack(
                [
                    torch.cat(
                    [
                        final_target_mask[i, -pad_lens[i]:],
                        final_target_mask[i, :embeds_lens[i]],
                    ]
                ) if pad_lens[i] != 0 else final_target_mask[i, :]
                for i in range(batch_size)
                ]
            )

        assert (final_attention_mask.size() ==
                targets.size())  # bsz x (1 + s1 + s2)
        return inputs_embeds, targets, final_attention_mask, final_target_mask ,{
            'input_ids': input_ids,
            'target_ids': target_ids,
        }


    def process_json_inputs(self, inputs, generation_mode:bool = False):
        """process json inputs

        :param dict inputs: input dict
        :return dict: processed input dict
        """
        assert (self.vision_type == inputs["vision_type"]
                ), "{} expected but {} given".format(self.vision_type,
                                                    inputs["vision_type"])
        task_type = inputs["task_type"]
        output_texts = inputs["output_texts"]
        vision_paths = inputs['vision_paths']
        vision_files = inputs['vision_files']
        path_and_files = zip(vision_paths, vision_files)
        vision_pils = []
        vision_names = []
        vision_feats = []
        for paths, files in path_and_files:
            assert len(paths) == len(files), "paths and files should have the same length, but got {} and {}".format(len(paths), len(files))
            if len(paths) == 0:
                continue
            vision_names.extend(paths)
            # assert all vision_file is not None, or all vision_file is None
            assert all([f is not None for f in files]) or all([f is None for f in files]), "all vision_file should be not None or all vision_file should be None"
            if files[0] is None:
                files = [load_image(file,return_np=True,image_size=self.im_size)[1] for file in paths]
            else:
                raise NotImplementedError("vision_files is not None")
            vision_pils.extend(files)   
        feats = self.visual_encoder(vision_pils)
        vision_feats.extend(feats)
        self.vision_pils = vision_pils
        self.vision_feats = vision_feats
        if len(vision_feats) > 0:
            adapted_feats = self.adapter(feats)
        vision_instances = [
            {
                'name':vision_names[i],
                'embed':adapted_feats[i]
            } for i in range(len(vision_names))
        ]
        self.get_vision_instances_generator(vision_instances)
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len,
            self.vision_type,  
            num_vision_tokens=self.num_vision_token,
            num_task_tokens=self.task_token_size, 
            template = self.conv_template, 
            gpt_prefix=self.gpt_prefix,
            generation_mode=generation_mode
        )
        inputs_embeds, targets, attention_mask,target_mask, info = self.prompt_wrap(
            input_ids,
            target_ids,
            attention_mask,
            self.use_system,
            task_type,
            padding_right= not generation_mode,
        )
        
        target_ids[target_ids==VISION_CONTROL_ID] = INVALID_TOKEN_ID # to avoid calculating loss on vision tokens
        return inputs_embeds, targets, attention_mask,target_mask, info

    def forward(self, inputs):
        if not hasattr(self, 'iter_num'):
            self.iter_num = 0
            self.output_interval = 50 # set a interval to output log
            os.makedirs(os.path.join(self.log_path, 'images'), exist_ok=True) # we automatically save a decoded (training) image and original image every output_interval
        inputs_embeds, targets, attention_mask, target_mask, info = self.process_json_inputs(
            inputs)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
            use_cache=not self.use_flash_attn,
        )
        loss = outputs.loss
        if self.vision_loss is not None:
            assert self.deadapter is not None, "deadapter is None"
            hidden_states = outputs.hidden_states            
            vision_embeds = self.deadapter(hidden_states,adapter=self.adapter)
            vision_loss = self.vision_loss(vision_embeds, self.vision_feats, self.vision_pils, target_mask, self.decoder,self.adapter) 
            loss += vision_loss # text cross entropy loss + vision loss
        else:
            vision_loss = torch.Tensor([0.]).to(self.device)
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:,
                                                            1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(
            torch.long)  # [B*S]
        valid_mask = get_valid_mask(labels).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        if self.iter_num % self.output_interval == 0:
            valid_mask = get_valid_mask(labels)
            if deepspeed.comm.get_rank() == 0:
                if self.vision_loss is not None:
                    print('\nvision_loss: ', vision_loss.item(), 'loss: ', loss.item())
                    if self.decoder is not None:
                        for k in range(len(self.vision_feats)):
                            idx = target_mask == k
                            if idx.sum() > 0:
                                vision_embed = vision_embeds[idx]
                                original_image:np.ndarray = self.vision_pils[k].transpose(1,2,0)
                                break
                        vision_embed = vision_embed.unsqueeze(0) # 1 x num_vision_token x embed_dim
                        decoded_vision = self.decoder(vision_embed.detach().clone())[0].clamp(0, 1)
                        decoded_vision = decoded_vision.permute(1,2,0).float().cpu().detach().numpy()
                        decoded_vision = (decoded_vision * 255).astype(np.uint8)
                        decoded_image = Image.fromarray(decoded_vision)
                        decoded_image.save(os.path.join(self.log_path,'images', 'decoded_image_{}.png'.format(self.iter_num)))
                        original_image = Image.fromarray(original_image)
                        original_image.save(os.path.join(self.log_path,'images', 'original_image_{}.png'.format(self.iter_num)))
                decoded_tokens = self.llama_tokenizer.batch_decode(
                    chosen_tokens[valid_mask], skip_special_tokens=False)
                tmp_labels = labels.detach().clone()
                decoded_labels = self.llama_tokenizer.batch_decode(
                    tmp_labels[valid_mask], skip_special_tokens=False)
                valid_gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1))
                right_tokens = chosen_tokens.reshape(-1)[valid_gen_acc]
                right_labels = labels.reshape(-1)[valid_gen_acc]
                decode_right_tokens = self.llama_tokenizer.batch_decode(
                    right_tokens, skip_special_tokens=False)
                decode_right_labels = self.llama_tokenizer.batch_decode(
                    right_labels, skip_special_tokens=False)
                with open(os.path.join(self.log_path, 'log.txt'), 'a') as f:
                    f.write('[!] iter_num: {}\n'.format(self.iter_num))
                    f.write('[!] decoded_labels: {}\n'.format(decoded_labels))
                    f.write('[!] decoded_tokens: {}\n'.format(decoded_tokens))
                    f.write('[!] decode_right_tokens: {}\n'.format(decode_right_tokens))
                    f.write('[!] decode_right_labels: {}\n'.format(decode_right_labels))
        self.iter_num += 1
        return (loss,vision_loss), gen_acc

    def prepare_generation_embedding(self, inputs):
        """prepare for generation

        :param class inputs: model
        :return Dict: generation input
        """
        inputs_embeds, _, embeds_mask, target_mask, _ = self.process_json_inputs(
            inputs, generation_mode=True)


        return inputs_embeds, embeds_mask, target_mask
    @torch.no_grad()
    def generate_image_only(self,data):
        """
        This function is used to generate only images, without "<Img>" and "</Img>" Tokens. We clear and replace the generation context to be 'Assistant: <Img>' so the next num_vision_token tokens will be vision tokens. After generating num_vision_token tokens, we directly stop.

        Though slighly different from the generation process in the paper, this function can skip the generation of "<Img>" token. As "<Img>" is tokenized differently in different LLMs, you will need to set different token start criteria for different LLMs and this function avoids this trouble.
        """
        sov = VISION_TAGS["sov"][self.vision_type]
        for k in range(len(data['output_texts'])):
            data['output_texts'][k][-1]['value'] = sov # replace eov with sov
        inputs_embeds, targets, attention_mask, target_mask, _ = self.process_json_inputs(
            data ,generation_mode=True
        )
        bsz = len(data['vision_paths'])
        generated_imgs = []
        img_tensors = torch.zeros(bsz, self.num_vision_token,self.visual_hidden_dim, dtype=inputs_embeds.dtype).to(self.device)
        for i in range(self.num_vision_token):
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                labels=targets,
                use_cache=not self.use_flash_attn,
            )
            hidden_states = outputs.hidden_states
            predict_img_embeds = self.deadapter(hidden_states[:, -1, :],adapter = self.adapter)
            img_tensors[:, i, :] = predict_img_embeds
            nxt_embeds = self.adapter(predict_img_embeds)
            inputs_embeds = torch.cat([inputs_embeds, nxt_embeds.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones([attention_mask.size()[0], 1], dtype=attention_mask.dtype).to(self.device)], dim=1)
            target_mask = torch.cat([target_mask, INVALID_TOKEN_ID * torch.ones([target_mask.size()[0], 1], dtype=target_mask.dtype).to(self.device)], dim=1)
            targets = torch.cat([targets, INVALID_TOKEN_ID * torch.ones([targets.size()[0], 1], dtype=targets.dtype).to(self.device)], dim=1)
        logits = outputs.logits
        predict_tokens = torch.max(logits, dim=-1)[1] #bsz x s1
        predict_tokens = predict_tokens[:, -self.num_vision_token - 1: - 1] #bsz x num_vision_token
        text_tokens = self.llama_tokenizer.batch_decode(
            predict_tokens, skip_special_tokens=False)
        img_tensors = self.decoder(img_tensors)
        for k in range(bsz):
            generated_imgs.append(img_tensors[k])
        return img_tensors, text_tokens
    def generate(self, inputs):
        """
        inputs = {
            'image_paths': optional,
            'mode': generation mode,
            'prompt': human input prompt,
            'max_new_tokens': generation length,
            'top_p': top_p,
            'temperature': temperature
            'modality_embeds': None or torch.tensor
            'modality_cache': save the image cache
        }
        """
        #if self.add_tokens and (not hasattr(self, 'embedding_updated') or not self.embedding_updated):
        #    self._update_token_embeddings()
        input_embeds, input_masks, _  = self.prepare_generation_embedding(
            inputs['data'])
        #try decode input_embeds: bsz x seq_len x embed_dim, embed_token_weight: vocab_size x embed_dim
        embed_token_weights = self.llama_model.model.model.embed_tokens.weight
        logits = torch.matmul(input_embeds, embed_token_weights.transpose(0,1)) # bsz x seq_len x vocab_size
        input_ids = torch.argmax(logits,dim=-1)
        #decode
        input_text = self.llama_tokenizer.batch_decode(input_ids,skip_special_tokens=False)
        stopping_criteria = StoppingCriteriaList([
            LM4LVStoppingCriteria([[2277, 29937], [835], [29871, 13], [1, 2]], input_embeds)
        ]  # TODO: different template has corresponding end signal
        )
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            max_new_tokens=inputs["max_new_tokens"],
            top_k=inputs["top_k"],
            temperature=inputs["temperature"],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.llama_tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        return output_text
