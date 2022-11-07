# IDK-MRC

Code and dataset for EMNLP 2022 paper titled ["IDK-MRC: Unanswerable Questions for Indonesian Machine Reading Comprehension"](https://arxiv.org/abs/2210.13778).

## Dataset Description
I(n)dontKnow-MRC (IDK-MRC) is an Indonesian Machine Reading Comprehension dataset that covers answerable and unanswerable questions. Based on the combination of the existing answerable questions in TyDiQA, the new unanswerable question in IDK-MRC is generated using a question generation model and human-written question. Each paragraph in the dataset has a set of answerable and unanswerable questions with the corresponding answer.

Besides IDK-MRC dataset, several baseline datasets also provided:
1. Trans SQuAD : machine translated SQuAD 2.0 (Muis and Purwarianti, 2020)
2. TyDiQA: Indonesian answerable questions set from the TyDiQA-GoldP (Clark et al., 2020)
3. Model Gen: TyDiQA + the unanswerable questions output from the question generation model
4. Human Filt: Model Gen dataset that has been filtered by human annotator

You can find all datasets on the `dataset` directory, or you can also load it using [NusaCrowd](https://indonlp.github.io/nusa-catalogue/card.html?idk_mrc).

## Running Models

### Environment Setup
We encourage you to create a virtual environment using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) by running this script:
```bash
$ conda create --n "<your_env_name>" python=3.8.13
$ conda activate <your_env_name>
```

Install all dependencies:
```bash
$ pip install -r requirements.txt
```

Then, run this python script to download stanza resources:
```python
import stanza
stanza.download('id')
```

### MRC Models
Coming soon

### Question Generation Models
Coming soon

## Citation
Please cite this paper if you use any code or dataset in this repository:
```
@misc{putri2022idk,
    doi = {10.48550/ARXIV.2210.13778},
    url = {https://arxiv.org/abs/2210.13778},
    author = {Putri, Rifki Afina and Oh, Alice},
    title = {IDK-MRC: Unanswerable Questions for Indonesian Machine Reading Comprehension},
    publisher = {arXiv},
    year = {2022},
}
```