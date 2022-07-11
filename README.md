# Evaluation for Motion Puzzle
We use the ST-GCN based recognition model of [Yan et al. 2018] to evaluate generative models in our [paper](https://arxiv.org/pdf/2202.05274.pdf).

## Prepare data
First, prepare the datasets needed for training and evaluation as following:
- Make `datasets` directory
- Download the datasets from the [link](https://www.dropbox.com/sh/nqlk21em963iujd/AAAPn7P0CKLwlmqigBXjIaGZa?dl=0)
- Put them in the `datasets` directory

The structure of `datasets` directory will look like:
```
datasets/
├─ preprocess_styletransfer_classify.npz
├─ styletransfer_classify.npz
├─ styletransfer_generate.npz
├─ styletransfer_stylized_aberman_0.npz
├─ styletransfer_stylized_holden_0.npz
├─ styletransfer_stylized_ours_0.npz
├─ ...

```
## Test pretrained model 
Prepare pretrained models as following:
- Make `output` directory
- Download the pretrained models from the [link](https://www.dropbox.com/sh/9tdmyvz22lnzmga/AABRZZZS3UpXO59QBorc-0L3a?dl=0)
- Put them in the `output` directory

The structure of `output` directory will look like:
```
output/
├─ SRA/
│  ├─ latest_checkpoint.pth
├─ CRA/
│  ├─ latest_checkpoint.pth
```

Finally, you can evaluate the generated results according to each criteria. Please refer to `base_options.py` or `eval_options.py` under `options` directory for model and evaluation specifications. 
```
python evaluate.py --experiment_name CRA --criteria content --load_latest --mode eval
# or
python evaluate.py --experiment_name SRA --criteria style --load_latest --mode eval
```


## Train from scratch
You can train your own classifier from scratch by specifing the classifying criteria, e.g., content or style.
- For training a content classifier
    ```
    python recognition.py --experiment_name [EXPERIMENT_NAME] --criteria content 
    ```
- For training a style classifier
    ```
    python recognition.py --experiment_name [EXPERIMENT_NAME] --criteria style  
    ```



