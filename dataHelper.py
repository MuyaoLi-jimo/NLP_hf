import datasets,transformers
from pathlib import Path
from utils import load_json_file,dump_json_file,load_jsonl,generate_uuid,dump_jsonl
from typing import Union
import random
from rich import print 

random.seed(2024)

DATASET_LIST = ["restaurant_sup","laptop_sup","acl_sup",'agnews_sup']
FEW_SHOT_DATASET_LIST = ["restaurant_fs","laptop_fs","acl_fs",'agnews_fs']

dataset_class_num = {
    'restaurant': 3,
    'laptop': 3,
    'acl': 6,
    'agnews':4,
}
dataset_classlabel = {
    'restaurant':["res_positive","res_neutral","res_negative"],
    'laptop':["lap_positive","lap_neutral","lap_negative"],
    'acl':['Uses', 'Future', 'CompareOrContrast','Motivation', 'Extends', 'Background'],
    'agnews':['World', 'Sports', 'Business', 'Sci/Tech']
}

SHOTS = 32
LEAST_SHOTS = 8

def get_dataset(dataset_name:Union[str,list], sep_token)->datasets.DatasetDict:
    '''
    dataset_name: str, the name of the dataset
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    dataset_list = []
    start_label = 0
    labels = []
    if isinstance(dataset_name,str):
        dataset_name = [dataset_name]
    assert all(s[-3:] == dataset_name[0][-3:] for s in dataset_name)  #要么都是fs，要么都是sub
    if isinstance(dataset_name,list):
        for dn in dataset_name:
            if dn in set(DATASET_LIST):
                ds = get_one_dataset(dataset_name=dn,sep_token=sep_token,start_label=start_label)
            elif dn in set(FEW_SHOT_DATASET_LIST):
                ds = get_one_dataset(dataset_name=dn.replace("fs","sup"),sep_token=sep_token,start_label=start_label)
                ds = get_few_shot_dataset(dn,ds,sep_token)
            else:
                raise ValueError(f"what is {dn}?")
            labels.extend(dataset_classlabel.get(dn.split("_")[0]))
            dataset_list.append(ds)
            start_label += dataset_class_num.get(dn.split("_")[0])
    else:
        raise TypeError("wrong type!")
    
    train_dataset = datasets.concatenate_datasets([ds["train"] for ds in dataset_list])
    test_datset = datasets.concatenate_datasets([ds["test"] for ds in dataset_list])
    #print(train_dataset[0])
    train_dataset = train_dataset.cast_column('label',datasets.ClassLabel(names=labels))
    test_datset = test_datset.cast_column('label',datasets.ClassLabel(names=labels))
    
    dataset = datasets.DatasetDict({"train":train_dataset,"test":test_datset})
    return dataset

def get_few_shot_dataset(daraset_name:str,dataset:datasets.DatasetDict,sep_token)->datasets.DatasetDict:
    dataset_path = Path(__file__).parent/"dataset"/"few_shots"/f"{daraset_name}_{sep_token}"
    if dataset_path.exists():
        return datasets.load_from_disk(dataset_path)
    labels = dataset["train"].unique("label")
    train_dataset = dataset["train"]
    new_train_dataset = []
    for label in labels:
        label_datset = random.sample(list(train_dataset.filter(lambda example: example['label']==label)),LEAST_SHOTS)
        new_train_dataset.extend(label_datset)
    new_test_datset = random.sample(list(dataset['test']), SHOTS)
    few_shot_dataset = datasets.DatasetDict({
        'train':datasets.Dataset.from_list(new_train_dataset),
        'test':datasets.Dataset.from_list(new_test_datset),
    })
    few_shot_dataset.save_to_disk(dataset_path)
    return few_shot_dataset

def get_one_dataset(dataset_name:Union[str,list], sep_token, start_label:int)->datasets.DatasetDict:
    
    dataset = None

    if dataset_name in {"restaurant_sup" , "laptop_sup"}:
        dataset = absa_dataset(dataset_name,sep_token)
    elif dataset_name == "acl_sup":
        dataset = aclarc_dataset(dataset_name,sep_token)
    elif dataset_name=="agnews_sup":
        dataset = agnews_datset(dataset_name,sep_token)
    #def adjust_label(example):
        #example["label"] = example["label"]+start_label
        #return example
    #dataset = dataset.map(adjust_label)
    return dataset

def absa_dataset(dataset_name,sep_token)->datasets.DatasetDict:
    dataset_path = None
    if dataset_name == "restaurant_sup":
        dataset_path = Path(__file__).parent/"dataset"/"raw"/"SemEval14-res"
    elif dataset_name == "laptop_sup":
        dataset_path = Path(__file__).parent/"dataset"/"raw"/"SemEval14-laptop"
    else:
        raise ValueError("false dataset_name")

    train_dataset_json = load_json_file( dataset_path  / "train.json")
    test_dataset_json = load_json_file( dataset_path  / "test.json")

    label2idx = {"positive":2,"neutral":1,"negative":0}
    def gen(dataset:dict):
       
        for value in dataset.values():
            text = value.get("sentence") + " "+ sep_token + " " + value.get("term")
            label = dataset_name[:3]+'_'+value.get("polarity")
            
            yield({"text":text,"label":label,"id":dataset_name + str(value.get("id"))})
    
    train_dataset = datasets.Dataset.from_generator(gen,gen_kwargs={"dataset":train_dataset_json})
    test_dataset = datasets.Dataset.from_generator(gen,gen_kwargs={"dataset":test_dataset_json})
    dataset = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})
    return dataset

def aclarc_dataset(dataset_name,sep_token)->datasets.DatasetDict:
    dataset_path = Path(__file__).parent/"dataset"/"raw"/"ACL_ARC"
    train_dataset_list = load_jsonl( dataset_path  / "train.jsonl")
    test_dataset_list = load_jsonl( dataset_path  / "test.jsonl")
    #for d in train_dataset_list:
    # d["id"] = generate_uuid()
    #for d in test_dataset_list:
    #   d["id"] = generate_uuid()
    #dump_jsonl(train_dataset_list,dataset_path  / "train.jsonl")
    #dump_jsonl(test_dataset_list,dataset_path  / "test.jsonl")
    label2idx = {'Uses': 0, 'Future': 1, 'CompareOrContrast': 2,
                    'Motivation': 3, 'Extends': 4, 'Background': 5}
    def gen(dataset:dict):
        for value in dataset:
            yield({"text":value.get("text"),"label":value.get("label"),"id":value.get("id")})

    train_dataset = datasets.Dataset.from_generator(gen,gen_kwargs={"dataset":train_dataset_list})
    test_dataset = datasets.Dataset.from_generator(gen,gen_kwargs={"dataset":test_dataset_list})
    dataset = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})
    #dataset.save_to_disk(Path(__file__).parent/"dataset"/dataset_name)
    return dataset

def agnews_datset(dataset_name,sep_token)->datasets.DatasetDict:
    #'World', 'Sports', 'Business', 'Sci/Tech'
    dataset_path = Path(__file__).parent/"dataset"/"raw"/f"ag_news"
    
    if dataset_path.exists():
        return datasets.load_from_disk(dataset_path)
    dataset = datasets.load_dataset("fancyzhx/ag_news",split="test")
    ids = [generate_uuid() for _ in range(dataset.shape[0])]
    dataset = dataset.add_column(name="id",column=ids)
    dataset = dataset.cast_column("label",datasets.Value(dtype='string', id=None))
    dataset = dataset.train_test_split(test_size=0.1, seed=2022)
    dataset.save_to_disk(dataset_path)
    exit()
    return dataset


if __name__ == "__main__":
    #dataset=datasets.load_dataset(r"ceval/ceval-exam",name="computer_network")
    #print(dataset["dev"])
    dataset = get_dataset(["agnews_sup"],"|")  
    
    print(dataset["train"].unique("label"))   
    print(dataset["train"].features)  
    print(dataset.shape) 
    
    
