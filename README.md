# Learning Quantile Functions for Temporal Point Processes with Recurrent Neural Splines

Python/Pytorch implementation of the paper "Learning Quantile Functions for Temporal Point Processes with Recurrent Neural Splines", Souhaib Ben Taieb (@ AISTATS 2022)

## Cite
Please cite our paper if you use the code in your own work
```
@article{
    sbt2021,
    title={Learning Quantile Functions for Temporal Point Processes with Recurrent Neural Splines},
    author={Souhaib {Ben Taieb}},
    journal={International Conference on Artificial Intelligence and Statistics (AISTATS)},
    year={2022},
}
```

## Usage

```
git clone https://github.com/bsouhaib/qf-tpp/
```

You can see the list of arguments with the following command:

```
cd qf-tpp/code
python train.py -h
```

You can try the following command as an example:

```
cd qf-tpp/code
python -u train.py  --dataset_name=taxi --method=RQS_EXP-crps_qapprox  --config=5 --max_epochs=100
```


## Acknowledgments

Our implementation is based on [ifl-tpp](https://github.com/shchur/ifl-tpp/tree/original-code/code) and [nsf](https://github.com/bayesiains/nsf). We thank the authors for sharing their code.

## Notes

- A
- B