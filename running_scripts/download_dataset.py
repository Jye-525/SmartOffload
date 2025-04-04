from datasets import load_dataset, concatenate_datasets


# load_dataset("gsm8k", "main", cache_dir="/vast/users/amaurya/huggingface/")


datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset_name in datasets:
    dataset = load_dataset('THUDM/LongBench', dataset_name, cache_dir="/vast/users/amaurya/huggingface/", split='test')
    