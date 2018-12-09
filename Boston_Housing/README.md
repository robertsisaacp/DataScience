# Housing Values in Suburbs of Boston

## Objective: 
* Train regression models for medv (Median value of owner-occupied homes in `$1000`'s) using:
    * OLR
    * Regularized regression:
        * Ridge
        * LASSO
        * Elastic net
* Examine prediction metrics such as ${R}^{2}$ and RMSE
* Perform hyperparameter tuning with the elastic net regularization, assess model performance by splittig off hold-out data to test as unseen data 

### Dataset
This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive [link](http://lib.stat.cmu.edu/datasets/boston).

## References

* This prompt was taken from DataCamp course Supervised Learning with scikit-learn [link](https://www.datacamp.com/home)
* Harrison, D. and Rubinfeld, D.L. (1978) Hedonic prices and the demand for clean air. J. Environ. Economics and Management 5, 81–102.
* Belsley D.A., Kuh, E. and Welsch, R.E. (1980) Regression Diagnostics. Identifying Influential Data and Sources of Collinearity. New York: Wiley.