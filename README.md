# Information-Theoretic Hierarchical Perception (ITHP)

[![GitHub stars](https://img.shields.io/github/stars/joshuaxiao98/ITHP.svg?style=social&label=Star)](https://github.com/joshuaxiao98/ITHP/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/joshuaxiao98/ITHP.svg?style=social&label=Fork)](https://github.com/joshuaxiao98/ITHP/network/members)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/joshuaxiao98/318e10f2a51f45faeb983715fae32581/ithp_test.ipynb)

This repository is the home of the official implementation code for the paper "Information-Theoretic Hierarchical Perception for Multimodal Learning."

## Overview

The ITHP model is inspired by neurological information processing models. It utilizes the information bottleneck (IB) method to establish connections across different modalities, constructing compact and informative latent states for information flow. The model's hierarchical structure facilitates incremental distillation of information, making it an innovative approach to multimodal learning.

![Model](./assets/Model.png)

## Getting Started

To explore the ITHP model:

1. Clone this repository:
   ```bash
   git clone https://github.com/joshuaxiao98/ITHP.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Within the `./datasets` directory, execute `download_datasets.sh` to acquire the MOSI and MOSEI datasets, as detailed [here](https://github.com/WasifurRahman/BERT_multimodal_transformer).

4. Initiate model training with:
   ```bash
   python train.py
   ```

Experiment with the ITHP model directly in your browser using Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/joshuaxiao98/318e10f2a51f45faeb983715fae32581/ithp_test.ipynb)

## Customization

To adapt the model to your needs:

- Modify `train.py` for changes to variables, loss functions, or outputs.
- The `max_seq_length` parameter can be reduced from the default `50` to conserve memory.
- Adjust `train_batch_size` to manage memory usage in relation to batch size.

## Citation

If you find this model useful in your research, please consider citing:

```bibtex
@article{
	(To be filled)
}
```