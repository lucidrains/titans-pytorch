<img src="./fig2.png" width="400px"></img>

<img src="./fig1.png" width="400px"></img>

## Titans - Pytorch

Unofficial implementation of [Titans](https://arxiv.org/abs/2501.00663) in Pytorch. Will also contain some explorations into architectures beyond their simple 1-4 layer MLP for the neural memory module, if it works well to any degree.

## Appreciation

- [Eryk](https://github.com/sentialx) for sharing his early experimental results with me, positive for 2 layer MLP

## Install

```bash
$ pip install titans-pytorch
```

## Usage

```python
import torch
from titans_pytorch import NeuralMemory

mem = NeuralMemory(
    dim = 384,
    chunk_size = 64 # set to smaller chunk size for better perf on smaller sequence lengths (but more memory usage)
).cuda()

seq = torch.randn(2, 1024, 384).cuda()
retrieved, mem_state = mem(seq)

assert seq.shape == retrieved.shape
```

A transformer with the `MAC` configuration can be used as

```python
import torch
from titans_pytorch import MemoryAsContextTransformer

transformer = MemoryAsContextTransformer(
    num_tokens = 256,
    dim = 256,
    depth = 2,
    segment_len = 128,              # local attention window size
    num_persist_mem_tokens = 4,
    num_longterm_mem_tokens = 16,
)

token_ids = torch.randint(0, 256, (1, 1023))

loss = transformer(token_ids, return_loss = True) # (1, 1023, 256)
loss.backward()

# after much training

sampled = transformer.sample(token_ids[:, :4], 512)
```

## Experiments

```bash
$ pip install .[examples]
```

Then modify `train_mac.py` and run it to query nature

```bash
$ python train_mac.py
```

## Citations

```bibtex
@inproceedings{Behrouz2024TitansLT,
    title   = {Titans: Learning to Memorize at Test Time},
    author  = {Ali Behrouz and Peilin Zhong and Vahab S. Mirrokni},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:275212078}
}
```

```bibtex
@article{Sun2024LearningT,
    title   = {Learning to (Learn at Test Time): RNNs with Expressive Hidden States},
    author  = {Yu Sun and Xinhao Li and Karan Dalal and Jiarui Xu and Arjun Vikram and Genghan Zhang and Yann Dubois and Xinlei Chen and Xiaolong Wang and Oluwasanmi Koyejo and Tatsunori Hashimoto and Carlos Guestrin},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2407.04620},
    url     = {https://api.semanticscholar.org/CorpusID:271039606}
}
```

```bibtex
@inproceedings{Yang2024GatedDN,
    title   = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
    author  = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:274598177}
}
```

```bibtex
@inproceedings{Nguyen2024TurningUT,
    title   = {Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs},
    author  = {Minh Nguyen and Andrew Baker and Clement Neo and Allen Roush and Andreas Kirsch and Ravid Shwartz-Ziv},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:270870613}
}
```

```bibtex
@article{Zhu2024HyperConnections,
    title   = {Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Zihao Huang and Yutao Zeng and Yunyao Mao and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2409.19606},
    url     = {https://api.semanticscholar.org/CorpusID:272987528}
}
```

```bibtex
@article{Zhou2024ValueRL,
    title   = {Value Residual Learning For Alleviating Attention Concentration In Transformers},
    author  = {Zhanchao Zhou and Tianyi Wu and Zhiyun Jiang and Zhenzhong Lan},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2410.17897},
    url     = {https://api.semanticscholar.org/CorpusID:273532030}
}
```

```bibtex
@software{Kyrylov_Accelerated_Scan_2024,
    author  = {Kyrylov, Volodymyr},
    doi     = {10.5281/zenodo.10600962},
    title   = {Accelerated Scan},
    version = {0.1.2},
    year    = {2024}
}
```

```bibtex
@misc{wang2025testtimeregressionunifyingframework,
    title   = {Test-time regression: a unifying framework for designing sequence models with associative memory},
    author  = {Ke Alexander Wang and Jiaxin Shi and Emily B. Fox},
    year    = {2025},
    eprint  = {2501.12352},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url = {https://arxiv.org/abs/2501.12352},
}
```
