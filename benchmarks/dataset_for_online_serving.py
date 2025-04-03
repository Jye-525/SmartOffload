
import random, os
import json
import numpy as np
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset
from tqdm.asyncio import tqdm
from longbench.longbench_data import load_lb_datasets, load_lb_v2_dataset

############################ Real Dataset ###########################
def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int, None]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset

def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int, None]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, str, int, int, None]] = []
    for _ in range(num_requests):
        num_lines_needed = num_input_lines - num_prefix_lines
        sampled_lines = "".join(prefix_lines +
                                random.choices(poem_lines, k=num_lines_needed))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len, None))

    return sampled_requests


def sample_longbench_v2_requests(
    dataset_path: str,
    num_requests: int,
    max_input_len: int,
    max_output_len: int,
    tokenizer: PreTrainedTokenizerBase,
    random_seed: int = 1234,
) -> List[Tuple[str, int, int, None]]:
    # load longbench-v2 dataset
    template_0shot, datasets = load_lb_v2_dataset(random_seed=random_seed, cache_dir=dataset_path)
    # random.shuffle(dataset)
    filtered_dataset: List[Tuple[str, int, int, None]] = []
    for json_obj in tqdm(datasets):
        if len(filtered_dataset) == num_requests:
            break
        
        context = json_obj['context']
        template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', json_obj['question'].strip()).replace('$C_A$', json_obj['choice_A'].strip()).replace('$C_B$', json_obj['choice_B'].strip()).replace('$C_C$', json_obj['choice_C'].strip()).replace('$C_D$', json_obj['choice_D'].strip())
        
        # truncate the prompt to make it fit within the max_input_len
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_input_len:
            # Truncate the input_ids, we usually truncate the middle part
            input_ids = input_ids[:max_input_len//2] + input_ids[-max_input_len//2:]
            print(f"input_ids len={len(input_ids)}")
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        # Verify the prompt tokenizes back to the correct length
        prompt_len = len(tokenizer.encode(prompt))
        assert prompt_len <= max_input_len, f"Prompt length exceeds max_input_len: {prompt_len} > {max_input_len}"
        filtered_dataset.append((prompt, prompt_len, max_output_len, None))
        
    return filtered_dataset 

def sample_longbench_requests(
    dataset_path: str,
    subtasks: List[str],
    num_requests: int,
    max_input_len: int,
    max_output_len: Optional[int],
    tokenizer: PreTrainedTokenizerBase,
    random_seed: int = 1234
) -> List[Tuple[str, int, int, None]]:
    # Load all the longbench dataset 
    longbench_dataset = load_lb_datasets(random_seed=random_seed, tasks=subtasks, cache_dir=dataset_path)
    # filter the required number of request from the dataset
    filtered_dataset = []
    for json_object in tqdm(longbench_dataset):
        if len(filtered_dataset) == num_requests:
            break
        
        prompt_format = json_object['prompt_format'] 
        format_args = {k: v for k, v in json_object.items() if k not in ["prompt_format", "max_gen_len"]}
        prompt = prompt_format.format(**format_args)
        
        # tokenize the prompt
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_input_len:
            # Truncate the input_ids, we usually truncate the middle part
            input_ids = input_ids[:max_input_len//2] + input_ids[-max_input_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Verify the prompt tokenizes back to the correct length
        prompt_len = len(tokenizer(prompt).input_ids)
        assert prompt_len <= max_input_len, f"Prompt length exceeds max_input_len: {prompt_len} > {max_input_len}" 
        output_len = json_object['max_gen_len'] if max_output_len is None else max_output_len
        filtered_dataset.append((prompt, prompt_len, output_len, None))
         
    print(f"Total number of samples in LongBench dataset: {len(longbench_dataset)}")
    
    return filtered_dataset   

########################### Synthetic Dataset ###########################
def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    prefix_token_ids = np.random.randint(0,
                                         tokenizer.vocab_size,
                                         size=prefix_len).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(prefix_token_ids +
                                  [(offsets[i] + i + j) % tokenizer.vocab_size
                                   for j in range(input_lens[i])])

        input_requests.append((prompt, int(prefix_len + input_lens[i]),
                               int(output_lens[i]), None))

    return input_requests

def sample_fixedlen_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    # offset = np.random.randint(0, tokenizer.vocab_size)
    offset = np.random.randint(0, tokenizer.vocab_size - 1)
    # get a prompt
    token_ids = [(offset + i) % tokenizer.vocab_size for i in range(input_len)]
    prompt = tokenizer.decode(token_ids)
    # Verify the prompt tokenizes back to the correct length
    verified_token_ids = tokenizer.encode(prompt)
    while len(verified_token_ids) != input_len:
        # Adjust by adding or removing tokens
        if len(verified_token_ids) > input_len:
            # Remove extra tokens by truncating the prompt
            prompt = tokenizer.decode(verified_token_ids[:input_len])
        else:
            # Append tokens to reach the desired length
            remaining = input_len - len(verified_token_ids)
            extra_tokens = [(offset + input_len + j) % tokenizer.vocab_size 
                           for j in range(remaining)]
            prompt = tokenizer.decode(token_ids + extra_tokens)
        
        # Re-verify
        verified_token_ids = tokenizer.encode(prompt)
        
     
    input_requests = []
    for i in range(num_prompts):
        input_requests.append((prompt, int(input_len),
                               int(output_len), None))

    return input_requests


def sample_gsm8k_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    random_seed: int,
    max_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset from huggingface datasets.
    dataset = load_dataset("gsm8k", "main", cache_dir=dataset_path)
    
    # shuffle the dataset
    dataset = dataset.shuffle(seed=random_seed)
     
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = f"Solve: {dataset[i]['question']}\nAnswer:"
        prompt_token_ids = tokenizer(prompt).input_ids
        output_len = dataset[i]['answer']
        output_token_ids = tokenizer(output_len).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(output_token_ids) if max_output_len is None else max_output_len
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset