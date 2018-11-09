# tinygbt-ruby
This is a minimal pure Ruby implementation of the Gradient Boosted Trees algorithm, based on the original project: https://github.com/lancifollia/tinygbt

Currently supports L2 regression and binary classification (Log Loss function)

### Run from command line: 

> ruby -I . ./lib/example.rb

### Features

- Regression with L2 loss 
- Binary Classification with Logarithmic Loss (cross-entropy) support 

### Loss functions notes

| Task | target | Loss Function | gradient | hessian |
| --- | --- | --- | --- | --- |
| Regression | continuous | (y-p)^2 | 2*(y-p) | 2 (const)
| Classification | {0,1} | -(y log(p) + (1 - y) log(1 - p)) | p-y | p*(1-p)

[How to find the 1st and 2nd derivates of Log Loss function](https://stats.stackexchange.com/questions/231220/how-to-compute-the-gradient-and-hessian-of-logarithmic-loss-question-is-based)

### References

- [1] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
- [2] G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. 2017.

### License

MIT
