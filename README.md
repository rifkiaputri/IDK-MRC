# IDK-MRC

Code and dataset for EMNLP 2022 paper titled ["IDK-MRC: Unanswerable Questions for Indonesian Machine Reading Comprehension"](https://aclanthology.org/2022.emnlp-main.465/).

## Dataset Description
I(n)dontKnow-MRC (IDK-MRC) is an Indonesian Machine Reading Comprehension dataset that covers answerable and unanswerable questions. Based on the combination of the existing answerable questions in TyDiQA, the new unanswerable question in IDK-MRC is generated using a question generation model and human-written question. Each paragraph in the dataset has a set of answerable and unanswerable questions with the corresponding answer.

Besides IDK-MRC dataset, several baseline datasets also provided:
1. Trans SQuAD : machine translated SQuAD 2.0 (Muis and Purwarianti, 2020)
2. TyDiQA: Indonesian answerable questions set from the TyDiQA-GoldP (Clark et al., 2020)
3. Model Gen: TyDiQA + the unanswerable questions output from the question generation model
4. Human Filt: Model Gen dataset that has been filtered by human annotator

You can find all datasets on the `dataset` directory. You can also load it using [NusaCrowd](https://indonlp.github.io/nusa-catalogue/card.html?idk_mrc) or [HuggingFace](https://huggingface.co/datasets/rifkiaputri/idk-mrc).

## Running Models

### Environment Setup
We encourage you to create a virtual environment using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) by running this script:
```bash
conda create --n "<your_env_name>" python=3.8.13
conda activate <your_env_name>
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

Then, run this python script to download stanza resources:
```python
import stanza
stanza.download('id')
```

### Available Models
3 MRC models finetuned on IDK-MRC dataset can be downloaded from HuggingFace:
- IndoBERT (base, uncased): [rifkiaputri/indobert-base-id-finetune-idk-mrc](https://huggingface.co/rifkiaputri/indobert-base-id-finetune-idk-mrc)
- m-BERT (base, cased): [rifkiaputri/mbert-base-id-finetune-idk-mrc](https://huggingface.co/rifkiaputri/mbert-base-id-finetune-idk-mrc)
- XLM-R (base, cased): [rifkiaputri/xlmr-base-id-finetune-idk-mrc](https://huggingface.co/rifkiaputri/xlmr-base-id-finetune-idk-mrc)

Also, 1 Unanswerable Question Generation (Unanswerable QG) model trained on translated SQuAD dataset can be downloaded from HuggingFace:
- mT5 (base, cased): [rifkiaputri/mt5-base-id-finetune-unans-qg](https://huggingface.co/rifkiaputri/mt5-base-id-finetune-unans-qg)

### Train Your Own Model
Run this following script to train MRC model:
```bash
python qa/run_qa.py \
    --train_path dataset/idk_mrc/train.json \
    --val_path dataset/idk_mrc/valid.json \
    --test_path dataset/idk_mrc/test.json \
    --epoch 10 \
    --model_name bert \
    --pretrain_name indobenchmark/indobert-base-p2 \
    --n_gpu 1 \
    --wb_name idk-mrc --wb_name_test idk-mrc-test \
    --out_path outputs \
    --out_name indobert_squad \
    --lang id \
    --uncased \
    --seed 42
```

Run this following script to train Unanswerable QG model:
```bash
python run_mt5.py \
    --train_path dataset/qg/train_id_aligned.json \
    --val_path dataset/qg/dev_id_aligned.json \
    --val_path_for_writing dataset/qg/dev_id_aligned_writing.json \
    --pretrain_name google/mt5-base \
    --qa_model outputs/qa/qgen-squad-id-trans/ \
    --n_gpu 1 \
    --wb_train_name mt5_squad_train \
    --wb_eval_name mt5_squad_ubleu_eval \
    --wb_eval_run_name mt5_base_id \
    --out_dir outputs/ \
    --cache_dir cache_dir/ \
    --seed 42 \
    --num_train_epochs 5 \
    --train_batch_size 8 \
    --max_seq_length 512 \
    --top_k 50 \
    --top_p 0.95 \
    --num_return_seq 10 \
    --do_sample
```

## Citation
Please cite this paper if you use any code or dataset in this repository:
```
@inproceedings{putri-oh-2022-idk,
    title = "{IDK}-{MRC}: Unanswerable Questions for {I}ndonesian Machine Reading Comprehension",
    author = "Putri, Rifki Afina  and
      Oh, Alice",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.465",
    pages = "6918--6933",
}
```
