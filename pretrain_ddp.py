
import os
import time
import json
import random
import argparse
import pickle

from tqdm import tqdm
import numpy as np
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from tokenizations import tokenization_bert
import transformers
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from utils import BaseDatasets


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def initialize_distributed(args):
    """Initialize torch.distributed."""

    device = args.rank % torch.cuda.device_count()
    print(f'rank = {args.rank} || local_rank = {args.local_rank}')
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    if args.world_size > 1:
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        dist.init_process_group(backend=args.distributed_backend,
                                world_size=args.world_size,
                                rank=args.rank,
                                init_method=init_method)
        dist.all_reduce(torch.zeros(1).cuda())
        suppress_output(args.rank == 0)


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='data/train.json', type=str, required=False, help='Train data')
    parser.add_argument('--eval_data_path', default='data/eval.json', type=str, required=False, help='Eval data')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='Train forloop nums')
    parser.add_argument('--perrank_batch_size', default=8, type=int, required=False, help='train batch size every rank')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--max_len', default=None, type=int, required=False, help='text max length')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='How long steps to print loss')
    parser.add_argument('--eval_step', default=100, type=int, required=False, help='How long steps to eval model')
    parser.add_argument('--save_model_interval_steps', default=100, type=int, required=False, help='How long steps to save model')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='gradient accumulation')
    parser.add_argument('--fp16', action='store_true', help='mixed precision')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='Model save dir')
    parser.add_argument('--pretrained_model', default='', type=str, required=True, help='Model start weights dir')
    parser.add_argument('--eval', action='store_true', help='Only eval')
    parser.add_argument('--add_prompt', default=0, type=int, required=False, help='prompt token nums')
    parser.add_argument('--add_pos', default=0, type=int, required=False, help='add pos id nums')
    parser.add_argument('--add_ner', default=0, type=int, required=False, help='add ner id nums')
    parser.add_argument('--add_cls_token', type=bool,  required=False, default=True, help='add_cls_token')
    parser.add_argument("--seed", default=42, type=int, required=False, help="seed")
    parser.add_argument('--local_rank', type=int, default=None, help='local rank passed from distributed launcher')
    parser.add_argument('--distributed-backend', default='nccl', help='which backend to use for distributed training. One of [gloo, nccl]')
    parser.add_argument('--frozen_pretrain', action='store_true', help='fix pretrained weights, only train prompt or pos or ner weights')
    parser.add_argument('--use_rotary_emb', action='store_true', help='use_rotary_emb')
    parser.add_argument('--use_position_emb', action='store_true', help='use_position_emb')
    parser.add_argument('--save_model', action='store_true', help='whethor save model')
    parser.add_argument("--skip_step", default=0, type=int, required=False, help="skip data step")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    args.cuda = torch.cuda.is_available()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    print(f'world_size: {args.world_size}')

    if args.world_size > 1:
        initialize_distributed(args)

    set_random_seed(args.seed)

    config_file = os.path.join(args.pretrained_model, 'config.json')
    config = transformers.modeling_gpt2.GPT2Config.from_json_file(config_file)
    config.add_pos = args.add_pos
    config.add_ner = args.add_ner
    config.add_prompt = args.add_prompt
    config.use_rotary_emb = args.use_rotary_emb
    config.use_position_emb = args.use_position_emb
    
    print('config:\n' + config.to_json_string())
    # Prompt token处的pos和ner的tag为2，因此全部的id为0,1,2，向量个数必须大于等于3.不然会报错。
    if config.add_prompt > 0 and config.add_pos:
        assert config.add_pos >= 3
    if config.add_prompt > 0 and config.add_ner:
        assert config.add_ner >= 3
    # if args.max_len is not None:
    #     max_len = args.max_len
    #     config.n_ctx = max_len
    # else:
    config.max_len = args.max_len
    max_len = args.max_len
    print(f'max_len: {max_len}')

    # vocab_file = os.path.join(args.pretrained_model, 'vocab.txt')
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='./vocab.txt')
    full_tokenizer.max_len = 999999

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    train_data_path = args.train_data_path
    eval_data_path = args.eval_data_path
    perrank_batch_size = args.perrank_batch_size
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model, config=config)
    if args.use_position_emb:
        div = config.max_len - model.transformer.wpe.weight.data.shape[0]
        if div > 0:
            extend_embed = model.transformer.wpe.weight.data[-1:].expand(div, config.n_embd)
            model.transformer.wpe.weight.data = torch.cat([model.transformer.wpe.weight.data, extend_embed], dim=0)

    eval_dataset = BaseDatasets(eval_data_path, 
                                tokenizer=full_tokenizer, 
                                add_pos=args.add_pos, 
                                add_ner=args.add_ner, 
                                add_prompt=args.add_prompt, 
                                max_len=max_len, 
                                add_cls_token=args.add_cls_token,
                                seed=args.seed,
                                rank=args.rank,
                                world_size=args.world_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=perrank_batch_size, shuffle=False, drop_last=True)
    eval_batch_nums =  len(eval_dataloader)
    print(f'rank: {args.rank} eval batch nums: {eval_batch_nums}')

    if not args.eval:
        train_dataset = BaseDatasets(train_data_path, 
                                tokenizer=full_tokenizer, 
                                add_pos=args.add_pos, 
                                add_ner=args.add_ner, 
                                add_prompt=args.add_prompt, 
                                max_len=max_len, 
                                add_cls_token=args.add_cls_token,
                                seed=args.seed,
                                rank=args.rank,
                                world_size=args.world_size)

        train_dataloader = DataLoader(train_dataset, batch_size=perrank_batch_size, shuffle=True, drop_last=True)
        train_batch_nums = len(train_dataloader)
        print(f'rank: {args.rank} train batch nums: {train_batch_nums}')

    
    model.train()
    model.to(device)
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # 冻结参数
    for name, params in model.named_parameters():
        if args.add_pos and 'pos.' in name:
            params.requires_grad = True
        elif args.add_ner and 'ner.' in name:
            params.requires_grad = True
        elif args.add_prompt and 'prompt.' in name:
            params.requires_grad = True
        else:
            if args.frozen_pretrain:
                params.requires_grad = False
            else:
                params.requires_grad = True
        print(name, params.shape, params.requires_grad)

    multi_gpu = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = DataParallel(model)
        model = DistributedDataParallel(model,
                                    device_ids=[args.rank],
                                    output_device=args.rank,
                                    find_unused_parameters=True)
        multi_gpu = True

    def evaluate(step, model, stop_num=None):
        losses = []
        print(f'Start evaluation...... multi_gpu: {multi_gpu}')
        start_time = time.time()
        for index,  batch in enumerate(eval_dataloader):
            model.eval()
            batch_inputs = batch['input_ids'].long().to(device)
            batch_labels = batch['labels'].long().to(device)
            batch_poss = batch['poss'].long().to(device)
            batch_ners = batch['ners'].long().to(device)

            outputs = model.forward(input_ids=batch_inputs, 
                                    pos_tags=batch_poss,
                                    ner_tags=batch_ners,
                                    labels=batch_labels)
            loss = outputs[0]
            if index % args.log_step == 0:
                print(f'Index: {index} loss: {loss.item()} take: {time.time() - start_time}s')
            if multi_gpu:
                loss = loss.mean()
            nan_mask = torch.isnan(loss)
            has_nan = torch.any(nan_mask).item()
            if has_nan:
                print(f'error: exist nan - {index}')
                continue
            losses.append(loss.item())
            if stop_num is not None and index == stop_num:
                break
                
        mean_loss = sum(losses) / len(losses)
        model.train()
        print(f'Finished evaluation. Cur step: {step} val loss: {mean_loss}.')
        return mean_loss

    if args.eval:
        eval_loss = evaluate('Unknow', model)
        exit(0)

    total_steps = int(args.epochs * train_batch_nums / args.gradient_accumulation)
    print('Total steps = {}'.format(total_steps))
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)
   
    def save_model(step, model, scheduler=None, optimizer=None):
        save_dir = os.path.join(output_dir, f'step{step}')
        print(f'Saving model to dir: {save_dir}')
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(save_dir)
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_dir,f'scheduler.pt'))
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))

    print('starting training')
    runed_steps = 0
    eval_max_loss = 10000.0
    start_time = time.time()
    for epoch in range(args.epochs):
        for step,  batch in enumerate(train_dataloader):
            if runed_steps < args.skip_step:
                runed_steps += 1
                continue
            batch_inputs = batch['input_ids'].long().to(device)
            batch_labels = batch['labels'].long().to(device)
            batch_poss = batch['poss'].long().to(device)
            batch_ners = batch['ners'].long().to(device)
            outputs = model.forward(input_ids=batch_inputs, 
                                    pos_tags=batch_poss,
                                    ner_tags=batch_ners,
                                    labels=batch_labels)
            loss, logits = outputs[:2]

            if multi_gpu:
                loss = loss.mean()

            if step % args.log_step == 0:
                print(f'Epoch: {epoch} step: {step}/{train_batch_nums} loss: {loss.item()} take: {time.time() - start_time}s')

            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
           
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (step + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            runed_steps += 1
            if runed_steps % args.eval_step == 0:
                eval_loss = evaluate(runed_steps, model)
                if  (args.save_model and runed_steps % args.save_model_interval_steps == 0 and eval_loss < eval_max_loss):
                    save_model(runed_steps, model)
        print(f'Epoch: {epoch} finished, take: {time.time() - start_time}s')
    print('Training finished')
    if args.save_model:
        save_model(runed_steps)

if __name__ == '__main__':
    main()
