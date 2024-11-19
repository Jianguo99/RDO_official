# Resolution invariant deep operator network for PDEs with complex geometries

This repository is the official implementation
of [Resolution invariant deep operator network for PDEs with complex geometries](https://arxiv.org/abs/2402.00825) at Journal of Computational Physics

## Preparing your dataset

1. Run data/step1_genereate_data.ipynb to generate the raw datasets
2. Run data/step2_split.ipynb to split the data to train/validation/test datasets

## Training neural operators

Producing the prediction sets:

```
python main_train_model.py
```

with the following arguments:

- lx: the length correlation.
- model: the name of the model.
- r: the resolution of input field.
- integral_type: the type of integral used in the model.
- seed: the random seed for reproducibility.
- gpu: the ID of the GPU to use.

## Citation

If you find this useful in your research, please consider citing:

    @article{huang2024resolution,
    title={Resolution invariant deep operator network for PDEs with complex geometries},
    author={Huang, Jianguo and Qiu, Yue},
    journal={arXiv preprint arXiv:2402.00825},
    year={2024}
    }

