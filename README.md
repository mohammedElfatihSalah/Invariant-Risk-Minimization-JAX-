# Invariant-Risk-Minimization-JAX-
Implementation of the paper [Invariant Risk Minimization (IRM)](https://arxiv.org/abs/1907.02893) in **JAX**, a learning paradigm to estimate invariant correlations across multiple training distributions. It 
learns a data representation such that the optimal classifier, on top of that data representation, matches for all training distributions.

## How to run the code?
First clone the repository by running the following
```
git clone https://github.com/mohammedElfatihSalah/Invariant-Risk-Minimization-JAX-.git
```
after that install the required dependencies,

```
sh dependencies.sh
```

## Data (Colored MNIST)

Which is the same MNIST image, but added color to it (either red or green) in a way that correlate with the image labels. so any algorithm purely minimizing training error will tend to
exploit the color. Such algorithms will fail at test time because the direction of the correlation is reversed.

## Results
Achieved similar results to the paper. I ran IRM vs ERM (Empirical Risk Minimization)in colored MNIST data, and the table
below summarizes the results:

| Tables        | Train Accuracy          | Test Accuracy  |
| ------------- |:-------------:| -----:|
| IRM      | 62% | 50%|
| ERM     | 85%      |   9% |
