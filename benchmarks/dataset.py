import os
import json
import random
import numpy as np
import dataclasses
import tqdm
from PIL import Image
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Optional, Tuple
from vllm.multimodal import MultiModalDataDict
from vllm.lora.request import LoRARequest


@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        prompt_len: The length of the prompt in tokens.
        expected_output_len: The expected length of the output in tokens.
        multi_modal_data: Optional dictionary containing multi-modal data (e.g.
            images).
        lora_request: Optional LoRARequest specifying the LoRA to use. 
    """
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[MultiModalDataDict] = None
    lora_request: Optional[LoRARequest] = None


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int,
    context_len: int,
)->List[SampleRequest]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Raise error if the sharegpt is not downloaded or the path is not valid.
    if not os.path.isfile(dataset_path) and dataset_path == "":
        raise ValueError("The dataset path is not valid.Please make sure the dataset is downloaded or the path is correct")

    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[SampleRequest] = []
    for data in tqdm(dataset,
                     total=len(filtered_dataset),
                     desc="sampling requests"):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = data[0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = data[1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append(SampleRequest(prompt=prompt,
                          prompt_len=prompt_len,
                          expected_output_len=output_len,
                          multi_modal_data=None,
                          lora_request=None))
    
    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.expected_output_len for x in filtered_dataset])}")
    return filtered_dataset


def sample_random_requests(
    input_len: int, 
    output_len: int, 
    num_prompts: int, 
    range_ratio: float, 
    tokenizer: PreTrainedTokenizerBase,
    use_real_data: bool = False, 
    dataset_path: Optional[str] = None
)->List[SampleRequest]:
    
    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    
    if not use_real_data:
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        input_requests: List[SampleRequest] = []
        for i in range(num_prompts):
            prompt = tokenizer.decode([(offsets[i] + i + j) % tokenizer.vocab_size
                                    for j in range(input_lens[i])])

            input_requests.append(SampleRequest(prompt=prompt,
                          prompt_len=int(input_lens[i]),
                          expected_output_len=int(output_lens[i]),
                          multi_modal_data=None,
                          lora_request=None))
        return input_requests
        
    else:
        assert dataset_path is not None and os.path.isfile(dataset_path), "The dataset path is not valid. Please make sure the SharedGPT dataset is downloaded or the path is correct"
        with open(dataset_path, encoding='utf-8') as f:
            dataset = json.load(f)
        
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data
            for data in dataset
            if len(data.get("conversations", data.get("conversation", []))) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data.get("conversations", data.get("conversation", []))[0]["value"],
                data.get("conversations", data.get("conversation", []))[1]["value"],
            )
            for data in dataset
        ]
        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[SampleRequest] = []
        for data in tqdm(dataset,
                     total=len(input_requests),
                     desc="sampling requests"):
            i = len(input_requests)
            if i == num_prompts:
                break

            # Tokenize the prompts and completions.
            prompt = data[0]
            prompt_token_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_token_ids)

            # Skip empty prompt
            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            prompt = tokenizer.decode(input_ids)
            
            input_requests.append(SampleRequest(prompt=prompt,
                          prompt_len=int(input_lens[i]),
                          expected_output_len=int(output_lens[i]),
                          multi_modal_data=None,
                          lora_request=None))

    print(f"#Input tokens: {np.sum([x.prompt_len for x in input_requests])}")
    print(f"#Output tokens: {np.sum([x.expected_output_len for x in input_requests])}")
    return input_requests
            
            

def get_dataset(args, tokenizer)->List[SampleRequest]:
    if args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
        )
    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            use_real_data=args.random_use_real_data,
            dataset_path=args.dataset_path,
        )
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    
    return input_requests