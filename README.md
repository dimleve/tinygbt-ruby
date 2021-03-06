# tinygbt-ruby
This is a minimal pure Ruby implementation of the Gradient Boosted Trees algorithm, based on the original project: https://github.com/lancifollia/tinygbt

Currently supports L2 regression and binary classification (Log Loss function)

### Gradient Boosted Trees algorithm: 

In the following post you can find some maths and algorithms behind the implementation:

https://medium.com/@dimleve/xgboost-mathematics-explained-58262530904a

### Run from command line: 

> cd lib <br />
> ruby example.rb

### Features

- Regression with L2 loss 
- Binary Classification with Logarithmic Loss (cross-entropy) support 

### Loss functions notes

| Task | target | Loss Function | gradient | hessian |
| --- | --- | --- | --- | --- |
| Regression | continuous | (y-p)^2 | 2*(y-p) | 2 (const)
| Classification | {0,1} | -(y log(p) + (1 - y) log(1 - p)) | p-y | p*(1-p)

[How to find the 1st and 2nd gradients of the Logarithmic Loss function](https://stats.stackexchange.com/questions/231220/how-to-compute-the-gradient-and-hessian-of-logarithmic-loss-question-is-based)

### Create GEM

Go to the base directory of the project and run:
>gem build tinygbt_ruby.gemspec <br />
>gem install ./tinygbt_ruby-0.1.0.gem <br />

And if you want to use tinygbt-ruby in you project just include it by:
> require 'tinygbt'

### References

- [1] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
- [2] G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. 2017.

### License

MIT
