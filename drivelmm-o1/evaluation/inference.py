import argparse
import itertools
import json
import math
import os
import random
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError('CUDA device is required for DriveLMM-o1 inference.')
    num_layers = {
        'InternVL2_5-1B': 24,
        'InternVL2_5-2B': 24,
        'InternVL2_5-4B': 32,
        'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48,
        'InternVL2_5-38B': 60,
        'InternVL2_5-78B': 80,
    }[model_name]
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f'language_model.base_model.model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.base_model.model.model.tok_embeddings'] = 0
    device_map['language_model.base_model.model.model.embed_tokens'] = 0
    device_map['language_model.base_model.model.model.rotary_emb'] = 0
    device_map['language_model.base_model.model.output'] = 0
    device_map['language_model.base_model.model.model.norm'] = 0
    device_map['language_model.base_model.model.lm_head'] = 0
    device_map[f'language_model.base_model.model.model.layers.{num_layers - 1}'] = 0
    return device_map


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def build_transform(input_size, pad2square=False):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = [T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img)]
    if pad2square:
        transforms.append(T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in mean))))
    transforms.extend([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return T.Compose(transforms)


def load_model_and_tokenizer(model_path, use_flash_attn=True):
    from transformers import AutoModel, AutoTokenizer

    device_map = split_model('InternVL2_5-8B')
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def build_mvpruner_config(pruning_ratio, log_pruning=False):
    base_view_budget = [0.06, 0.22, 0.10, 0.06, 0.10, 0.04]
    total = sum(base_view_budget)
    if abs(pruning_ratio - 0.75) < 1e-6:
        stage_keep_ratio = 0.3660254
        stage2_layer = 16
        attention_layer = 15
        budget_scale = 0.7333
    else:
        stage_keep_ratio = 0.167
        stage2_layer = 16
        attention_layer = 15
        budget_scale = 0.7419
    fastv_r = [x / total * budget_scale * 6 for x in base_view_budget]
    return {
        'fastv_k': 1,
        'fastv_r': fastv_r,
        'fastv_random': True,
        'farwaypoint': True,
        'get_texttoken_start': True,
        'prune_method': 'farway',
        'pruning_ratio': pruning_ratio,
        'stage_keep_ratio': stage_keep_ratio,
        'stage1_layer': 0,
        'stage2_layer': stage2_layer,
        'attention_layer': attention_layer,
        'tokens_per_view': 256,
        'num_views': 6,
        'log_pruning': log_pruning,
    }


class DriveLMMo1Dataset(torch.utils.data.Dataset):
    def __init__(self, data_jsonl, meta_json, image_root, prompt, input_size, max_samples=None):
        with open(data_jsonl, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        with open(meta_json, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        if max_samples is not None:
            self.data = self.data[:max_samples]
            self.meta = self.meta[:max_samples]
        if len(self.data) != len(self.meta):
            raise ValueError(f'data_jsonl and meta_json length mismatch: {len(self.data)} vs {len(self.meta)}')
        self.image_root = Path(image_root)
        self.prompt = prompt
        self.transform = build_transform(input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        meta = self.meta[idx]
        data_id = data['id']
        if meta['idx'] != data_id:
            raise ValueError(f'id mismatch at index {idx}: {data_id} vs {meta["idx"]}')
        images = []
        for rel_path in meta['image']:
            image_file = self.image_root / rel_path
            images.append(Image.open(image_file).convert('RGB'))
        pixel_values = torch.stack([self.transform(image) for image in images])
        question = data['conversations'][0]['value'].strip()
        answer = data['conversations'][1]['value'].strip()
        return {
            'question': self.prompt + '\n<image>\n' + question,
            'pixel_values': pixel_values,
            'answer': answer,
            'data_id': data_id,
        }


def collate_fn(batches):
    return (
        torch.cat([item['pixel_values'] for item in batches], dim=0),
        [item['question'] for item in batches],
        [item['answer'] for item in batches],
        [item['data_id'] for item in batches],
    )


class InferenceSampler(torch.utils.data.Sampler):
    def __init__(self, size, rank=0, world_size=1):
        self._size = int(size)
        shard_size = self._size // world_size
        left = self._size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def maybe_init_distributed():
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    return rank, world_size


def gather_outputs(outputs, rank, world_size):
    if world_size == 1:
        return outputs
    torch.distributed.barrier()
    merged = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged, json.dumps(outputs))
    return list(itertools.chain.from_iterable(json.loads(item) for item in merged))


def parse_args():
    parser = argparse.ArgumentParser(description='Run MVPruner inference on DriveLMM-o1.')
    parser.add_argument('--model-path', default='ckpt/DriveLMMo1-MVPruner')
    parser.add_argument('--data-jsonl', default='DriveLMMo1_TEST.jsonl')
    parser.add_argument('--meta-json', default='DriveLMMo1_TEST.json')
    parser.add_argument('--image-root', required=True, help='nuScenes root containing samples/CAM_* folders')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--output-name', default='mvpruner_90')
    parser.add_argument('--pruning-ratio', type=float, choices=[0.9, 0.75], default=0.9)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-new-tokens', type=int, default=2000)
    parser.add_argument('--min-new-tokens', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--log-pruning', action='store_true')
    parser.add_argument('--no-flash-attn', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError('Only batch size 1 is supported.')
    random.seed(args.seed)
    rank, world_size = maybe_init_distributed()

    model, tokenizer = load_model_and_tokenizer(args.model_path, use_flash_attn=not args.no_flash_attn)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    model.config.fastv_config = build_mvpruner_config(args.pruning_ratio, args.log_pruning)

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print('Model path:', args.model_path)
        print('Data jsonl:', args.data_jsonl)
        print('Meta json:', args.meta_json)
        print('Image root:', args.image_root)
        print('Output dir:', args.output_dir)
        print('Output name:', args.output_name)
        print('MVPruner config:', model.config.fastv_config)
        print(f'[test] image_size: {image_size}')
        print(f'[test] template: {model.config.template}')

    prompt = (
        'When answering the question based on the provided image, follow a structured and logical reasoning process. '
        'Organize your response using the format, ensuring each step builds upon the previous one and clearly explains '
        'how the image(s) contribute to the solution. Your answer should be structured as Reasoning Steps: '
        '(step by step reasoning) Final Answer: (final answer) \n Question: '
    )
    dataset = DriveLMMo1Dataset(
        data_jsonl=args.data_jsonl,
        meta_json=args.meta_json,
        image_root=args.image_root,
        prompt=prompt,
        input_size=image_size,
        max_samples=args.max_samples,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset), rank=rank, world_size=world_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    outputs = []
    for pixel_values, questions, answers, data_ids in tqdm(dataloader, disable=rank != 0):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = {
            'num_beams': args.num_beams,
            'max_new_tokens': args.max_new_tokens,
            'min_new_tokens': args.min_new_tokens,
            'do_sample': args.temperature > 0,
            'temperature': args.temperature,
        }
        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=questions[0],
            generation_config=generation_config,
        )
        if rank == 0:
            print('data_id:', data_ids[0])
            print('pred:', pred)
        outputs.append({
            'question': questions[0],
            'answer': pred,
            'gt_answers': answers[0],
            'id': data_ids[0],
        })

    merged_outputs = gather_outputs(outputs, rank=rank, world_size=world_size)
    if rank == 0:
        output_path = Path(args.output_dir) / f'DriveLMMo1_{args.output_name}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_outputs, f, ensure_ascii=False, indent=4)
        print(f'Results saved to {output_path}')


if __name__ == '__main__':
    main()
