# Information-Theoretic Hierarchical Perception

This repository contains the official implementation code of the paper _Information-Theoretic Hierarchical Perception for Multimodal Learning_.



## Instruction

The Information-Theoretic Hierarchical Perception (ITHP) model is drawing inspiration from neurological models of information processing, we build the links between different modalities using the information bottleneck (IB) method. By leveraging the IB principle, ITHP constructs compact and informative latent states for information flow. The hierarchical architecture of ITHP enables incremental distillation of useful information. 

![Model](./assets/Model.png)



## Usage

1. Clone the repository to your local machine:
    ```bash
    git clone <the repository's URL>
    ```

2. Set up the environment
    ```bash
    pip install -r requirements.txt
    ```

3. To train the model, run the `train.py` script:

    ```bash
    python train.py
    ```


**Modifications**

>If you need to modify the variables, loss functions, or the outputs of the training, you'll need to adjust the code in `train.py`.

The factors that affect memory usage are:

- **`max_seq_length`**: The released models were trained with sequence lengths `50`, but you can fine-tune with a shorter max sequence length to save substantial memory. This is controlled by the `max_seq_length` flag in our example code.
- **`train_batch_size`**: The memory usage is also directly proportional to the batch size.



## Citation

You may cite us using:

```tex
<Anonymous>
```