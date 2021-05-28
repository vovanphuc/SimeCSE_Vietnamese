
#### Table of contents
1. [Introduction](#introduction)
2. [Pretrain model](#models)
3. [Using SimeCSE_Vietnamese with `sentences-transformers`](#sentences-transformers)
	- [Installation](#install1)
	- [Example usage](#usage1)
4. [Using SimeCSE_Vietnamese with `transformers`](#transformers)
	- [Installation](#install2)
	- [Example usage](#usage2)
# <a name="introduction"></a> SimeCSE_Vietnamese: Simple Contrastive Learning of Sentence Embeddings with Vietnamese

Pre-trained SimeCSE_Vietnamese models are the state-of-the-art of Sentence Embeddings with Vietnamese : 

 - SimeCSE_Vietnamese pre-training approach is based on [SimCSE](https://arxiv.org/abs/2104.08821) which optimizes the SimeCSE_Vietnamese pre-training procedure for more robust performance.
 - SimeCSE_Vietnamese encode input sentences using a pre-trained language model such as  [PhoBert](https://www.aclweb.org/anthology/2020.findings-emnlp.92/)
 - SimeCSE_Vietnamese works with both unlabeled and labeled data.

## Pre-trained models <a name="models"></a>


Model | #params | Arch.	 
---|---|---
[`VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`](https://huggingface.co/VoVanPhuc/sup-SimCSE-VietNamese-phobert-base) | 135M | base 
[`VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base`](https://huggingface.co/VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base) | 135M | base 


## <a name="sentences-transformers"></a> Using SimeCSE_Vietnamese with `sentences-transformers` 


### Installation <a name="install1"></a>
 -  Install `sentence-transformers`:
	
	- `pip install -U sentence-transformers`
	
 - Install `pyvi` to word segment:

	- `pip install pyvi`

### Example usage <a name="usage1"></a>

```python
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

sentences = ['Kẻ đánh bom đinh tồi tệ nhất nước Anh.',
          'Nghệ sĩ làm thiện nguyện - minh bạch là việc cấp thiết.',
          'Bắc Giang tăng khả năng điều trị và xét nghiệm.',
          'HLV futsal Việt Nam tiết lộ lý do hạ Lebanon.',
          'việc quan trọng khi kêu gọi quyên góp từ thiện là phải minh bạch, giải ngân kịp thời.',
          '20% bệnh nhân Covid-19 có thể nhanh chóng trở nặng.',
          'Thái Lan thua giao hữu trước vòng loại World Cup.',
          'Cựu tuyển thủ Nguyễn Bảo Quân: May mắn ủng hộ futsal Việt Nam',
          'Chủ ki-ốt bị đâm chết trong chợ đầu mối lớn nhất Thanh Hoá.',
          'Bắn chết người trong cuộc rượt đuổi trên sông.'
          ]

sentences = [tokenize(sentence) for sentence in sentences]
embeddings = model.encode(sentences)
```

## <a name="sentences-transformers"></a> Using SimeCSE_Vietnamese with `transformers` 

### Installation <a name="install2"></a>
 -  Install `transformers`:

	- `pip install -U transformers`

	
 - Install `pyvi` to word segment:

	- `pip install pyvi`

### Example usage <a name="usage2"></a>

```python
import torch
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize

PhobertTokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

sentences = ['Kẻ đánh bom đinh tồi tệ nhất nước Anh.',
          'Nghệ sĩ làm thiện nguyện - minh bạch là việc cấp thiết.',
          'Bắc Giang tăng khả năng điều trị và xét nghiệm.',
          'HLV futsal Việt Nam tiết lộ lý do hạ Lebanon.',
          'việc quan trọng khi kêu gọi quyên góp từ thiện là phải minh bạch, giải ngân kịp thời.',
          '20% bệnh nhân Covid-19 có thể nhanh chóng trở nặng.',
          'Thái Lan thua giao hữu trước vòng loại World Cup.',
          'Cựu tuyển thủ Nguyễn Bảo Quân: May mắn ủng hộ futsal Việt Nam',
          'Chủ ki-ốt bị đâm chết trong chợ đầu mối lớn nhất Thanh Hoá.',
          'Bắn chết người trong cuộc rượt đuổi trên sông.'
          ]

sentences = [tokenize(sentence) for sentence in sentences]

inputs = PhobertTokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
```
## Quick Start

[Open In Colab](https://colab.research.google.com/drive/12__EXJoQYHe9nhi4aXLTf9idtXT8yr7H?usp=sharing)

## Citation


	@article{gao2021simcse,
	   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
	   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
	   journal={arXiv preprint arXiv:2104.08821},
	   year={2021}
	}

    @inproceedings{phobert,
    title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
    author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
    year      = {2020},
    pages     = {1037--1042}
    }
