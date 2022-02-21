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

- If you are using MLE, i.e. logarithmic score (LogS), and since our code is based on [ifl-tpp](https://github.com/shchur/ifl-tpp/tree/original-code/code), please read [this comment](https://github.com/shchur/ifl-tpp/blob/master/README.md#mistakes-in-the-old-version).
- By default, we compute the CRPS by numerically approximating the integral for faster computations (e.g., RQS_EXP-crps_qapprox_100 or RQS_EXP-crps_qapprox_200)
- Numerical errors can happen when computing/optimizing the closed-form expression of the CRPS. We adapted the code to deal with the ones we encountered so far. We believe there is still room for improvement both in terms of numerical stability and computational efficiency.
- To standardize the data (sequences), we apply a log transformation (see Appendix C3). This can be turned on/off with the boolean ``log_and_scaling``. You can use this transformation with RQS_EXP-crps_qapprox(\_n), but it is not yet implemented for the closed-form expressions, i.e. the CRPS and the expectation.

