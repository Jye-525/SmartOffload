import sys, os, json
from typing import List, Dict, Any, Tuple
from datasets import load_dataset, concatenate_datasets

def _load_and_process_dataset(dataset_name, cache_dir, prompt_format, max_gen_len):
    dataset = load_dataset('THUDM/LongBench', dataset_name, cache_dir=cache_dir, split='test')
    
    dataset = dataset.map(lambda x: {
        "dataset_name": dataset_name,
        "prompt_format": prompt_format,
        "max_gen_len": max_gen_len,
        **x
    })
    
    return dataset

def get_lb_dataset2prompt():
    cur_script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(cur_script_path)
    dataset2prompt = json.load(open(f"{base_dir}/config/dataset2prompt.json", "r"))
    return dataset2prompt

def get_lb_dataset2maxgenlen():
    cur_script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(cur_script_path)
    dataset2maxgenlen = json.load(open(f"{base_dir}/config/dataset2maxlen.json", "r"))
    return dataset2maxgenlen

def load_lb_datasets(cache_dir, tasks, random_seed=1234):
    # cur_script_path = os.path.abspath(__file__)
    # base_dir = os.path.dirname(cur_script_path)
    datasets=tasks
    # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
    #                 "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
    #                 "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = get_lb_dataset2prompt()
    dataset2maxgenlen = get_lb_dataset2maxgenlen()
    
    # Load and process each dataset
    processed_datasets = []
    for dataset_name in datasets:
        prompt_format = dataset2prompt[dataset_name]
        max_gen_len = dataset2maxgenlen[dataset_name]
        # Load and process the dataset
        dataset = _load_and_process_dataset(dataset_name, cache_dir, prompt_format, max_gen_len)
        processed_datasets.append(dataset)
        
    # Merge efficiently
    merged_dataset = concatenate_datasets(processed_datasets)
    # shuffle the dataset
    print("Before shuffle. first iterm: ", merged_dataset[0]['_id'])
    merged_dataset = merged_dataset.shuffle(seed=random_seed)
    print("After shuffle. first iterm: ", merged_dataset[0]['_id'])
    
    return merged_dataset

def load_lb_v2_dataset(cache_dir, random_seed=1234) -> Tuple[str, List[Dict[str, Any]]]:
    cur_script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(cur_script_path)
    template_0shot = open(f'{base_dir}/prompts/0shot.txt', encoding='utf-8').read()
    dataset = load_dataset('THUDM/LongBench-v2', cache_dir=cache_dir, split='train')
    dataset = dataset.shuffle(seed=random_seed)
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]    

    return template_0shot, data_all

def get_max_model_input_length(model_name: str) -> int:
    """
    Get the maximum input length for a given model name.
    """
    cur_script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(cur_script_path)
    print(f"base_dir: {base_dir} in get_max_model_input_length")
    model2maxlen = json.load(open(f"{base_dir}/config/model2maxlen.json", "r"))
    if model_name not in model2maxlen:
        raise ValueError(f"Model name {model_name} not found in the configuration.")
    max_input_length = model2maxlen[model_name]
    
    return max_input_length
    
  
    
if __name__ == '__main__':
    cache_dir = "/lus/eagle/projects/RECUP/jye/dataset/"
    # merged_dataset = load_lb_datasets()
    prompt_format, data_all = load_lb_v2_dataset(cache_dir)
    print(f"first data: {data_all[0]['_id']}")
    
     