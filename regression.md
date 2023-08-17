# Regression

Here we perform basic regresison analysis


## Using statsmodels

This follows the example in Kevin Sheppard's Introduction to Python (https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2021.pdf) Chapter 21.1 Regression. The statsmodels package has good documentation here:
https://www.statsmodels.org/stable/index.html




```python
import statsmodels.api as sm 
d = sm.datasets.statecrime.load_pandas()

```

The data are now loaded into `d`. That is a dataset and you can see the actual spreadsheet using the `.data` attribute.


```python
d.data



```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>violent</th>
      <th>murder</th>
      <th>hs_grad</th>
      <th>poverty</th>
      <th>single</th>
      <th>white</th>
      <th>urban</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama</th>
      <td>459.9</td>
      <td>7.1</td>
      <td>82.1</td>
      <td>17.5</td>
      <td>29.0</td>
      <td>70.0</td>
      <td>48.65</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>632.6</td>
      <td>3.2</td>
      <td>91.4</td>
      <td>9.0</td>
      <td>25.5</td>
      <td>68.3</td>
      <td>44.46</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <td>423.2</td>
      <td>5.5</td>
      <td>84.2</td>
      <td>16.5</td>
      <td>25.7</td>
      <td>80.0</td>
      <td>80.07</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <td>530.3</td>
      <td>6.3</td>
      <td>82.4</td>
      <td>18.8</td>
      <td>26.3</td>
      <td>78.4</td>
      <td>39.54</td>
    </tr>
    <tr>
      <th>California</th>
      <td>473.4</td>
      <td>5.4</td>
      <td>80.6</td>
      <td>14.2</td>
      <td>27.8</td>
      <td>62.7</td>
      <td>89.73</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>340.9</td>
      <td>3.2</td>
      <td>89.3</td>
      <td>12.9</td>
      <td>21.4</td>
      <td>84.6</td>
      <td>76.86</td>
    </tr>
    <tr>
      <th>Connecticut</th>
      <td>300.5</td>
      <td>3.0</td>
      <td>88.6</td>
      <td>9.4</td>
      <td>25.0</td>
      <td>79.1</td>
      <td>84.83</td>
    </tr>
    <tr>
      <th>Delaware</th>
      <td>645.1</td>
      <td>4.6</td>
      <td>87.4</td>
      <td>10.8</td>
      <td>27.6</td>
      <td>71.9</td>
      <td>68.71</td>
    </tr>
    <tr>
      <th>District of Columbia</th>
      <td>1348.9</td>
      <td>24.2</td>
      <td>87.1</td>
      <td>18.4</td>
      <td>48.0</td>
      <td>38.7</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>612.6</td>
      <td>5.5</td>
      <td>85.3</td>
      <td>14.9</td>
      <td>26.6</td>
      <td>76.9</td>
      <td>87.44</td>
    </tr>
    <tr>
      <th>Georgia</th>
      <td>432.6</td>
      <td>6.0</td>
      <td>83.9</td>
      <td>16.5</td>
      <td>29.3</td>
      <td>61.9</td>
      <td>65.38</td>
    </tr>
    <tr>
      <th>Hawaii</th>
      <td>274.1</td>
      <td>1.8</td>
      <td>90.4</td>
      <td>10.4</td>
      <td>26.3</td>
      <td>26.9</td>
      <td>71.46</td>
    </tr>
    <tr>
      <th>Idaho</th>
      <td>238.5</td>
      <td>1.5</td>
      <td>88.4</td>
      <td>14.3</td>
      <td>19.0</td>
      <td>92.3</td>
      <td>50.51</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>618.2</td>
      <td>8.4</td>
      <td>86.4</td>
      <td>13.3</td>
      <td>26.0</td>
      <td>72.5</td>
      <td>79.97</td>
    </tr>
    <tr>
      <th>Indiana</th>
      <td>366.4</td>
      <td>5.3</td>
      <td>86.6</td>
      <td>14.4</td>
      <td>24.5</td>
      <td>85.7</td>
      <td>59.17</td>
    </tr>
    <tr>
      <th>Iowa</th>
      <td>294.5</td>
      <td>1.3</td>
      <td>90.5</td>
      <td>11.8</td>
      <td>20.3</td>
      <td>92.3</td>
      <td>41.66</td>
    </tr>
    <tr>
      <th>Kansas</th>
      <td>412.0</td>
      <td>4.7</td>
      <td>89.7</td>
      <td>13.4</td>
      <td>22.8</td>
      <td>86.3</td>
      <td>50.17</td>
    </tr>
    <tr>
      <th>Kentucky</th>
      <td>265.5</td>
      <td>4.3</td>
      <td>81.7</td>
      <td>18.6</td>
      <td>25.4</td>
      <td>88.8</td>
      <td>40.99</td>
    </tr>
    <tr>
      <th>Louisiana</th>
      <td>628.4</td>
      <td>12.3</td>
      <td>82.2</td>
      <td>17.3</td>
      <td>31.4</td>
      <td>63.7</td>
      <td>61.33</td>
    </tr>
    <tr>
      <th>Maine</th>
      <td>119.9</td>
      <td>2.0</td>
      <td>90.2</td>
      <td>12.3</td>
      <td>22.0</td>
      <td>94.9</td>
      <td>26.21</td>
    </tr>
    <tr>
      <th>Maryland</th>
      <td>590.0</td>
      <td>7.7</td>
      <td>88.2</td>
      <td>9.1</td>
      <td>27.3</td>
      <td>60.2</td>
      <td>83.53</td>
    </tr>
    <tr>
      <th>Massachusetts</th>
      <td>465.6</td>
      <td>2.7</td>
      <td>89.0</td>
      <td>10.3</td>
      <td>25.0</td>
      <td>82.4</td>
      <td>90.30</td>
    </tr>
    <tr>
      <th>Michigan</th>
      <td>504.4</td>
      <td>6.3</td>
      <td>87.9</td>
      <td>16.2</td>
      <td>25.6</td>
      <td>79.9</td>
      <td>66.37</td>
    </tr>
    <tr>
      <th>Minnesota</th>
      <td>214.2</td>
      <td>1.5</td>
      <td>91.5</td>
      <td>11.0</td>
      <td>20.2</td>
      <td>87.4</td>
      <td>58.00</td>
    </tr>
    <tr>
      <th>Mississippi</th>
      <td>306.7</td>
      <td>6.9</td>
      <td>80.4</td>
      <td>21.9</td>
      <td>32.8</td>
      <td>59.6</td>
      <td>27.62</td>
    </tr>
    <tr>
      <th>Missouri</th>
      <td>500.3</td>
      <td>6.6</td>
      <td>86.8</td>
      <td>14.6</td>
      <td>25.3</td>
      <td>83.9</td>
      <td>56.61</td>
    </tr>
    <tr>
      <th>Montana</th>
      <td>283.9</td>
      <td>3.2</td>
      <td>90.8</td>
      <td>15.1</td>
      <td>20.3</td>
      <td>89.4</td>
      <td>26.49</td>
    </tr>
    <tr>
      <th>Nebraska</th>
      <td>305.5</td>
      <td>2.5</td>
      <td>89.8</td>
      <td>12.3</td>
      <td>20.9</td>
      <td>88.1</td>
      <td>53.78</td>
    </tr>
    <tr>
      <th>Nevada</th>
      <td>704.6</td>
      <td>5.9</td>
      <td>83.9</td>
      <td>12.4</td>
      <td>28.5</td>
      <td>76.2</td>
      <td>86.51</td>
    </tr>
    <tr>
      <th>New Hampshire</th>
      <td>169.5</td>
      <td>0.9</td>
      <td>91.3</td>
      <td>8.5</td>
      <td>19.5</td>
      <td>94.5</td>
      <td>47.34</td>
    </tr>
    <tr>
      <th>New Jersey</th>
      <td>311.3</td>
      <td>3.7</td>
      <td>87.4</td>
      <td>9.4</td>
      <td>25.8</td>
      <td>70.7</td>
      <td>92.24</td>
    </tr>
    <tr>
      <th>New Mexico</th>
      <td>652.8</td>
      <td>10.0</td>
      <td>82.8</td>
      <td>18.0</td>
      <td>29.1</td>
      <td>72.5</td>
      <td>53.75</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>385.5</td>
      <td>4.0</td>
      <td>84.7</td>
      <td>14.2</td>
      <td>30.2</td>
      <td>67.4</td>
      <td>82.66</td>
    </tr>
    <tr>
      <th>North Carolina</th>
      <td>414.0</td>
      <td>5.4</td>
      <td>84.3</td>
      <td>16.3</td>
      <td>26.3</td>
      <td>70.5</td>
      <td>54.88</td>
    </tr>
    <tr>
      <th>North Dakota</th>
      <td>223.6</td>
      <td>2.0</td>
      <td>90.1</td>
      <td>11.7</td>
      <td>18.2</td>
      <td>90.2</td>
      <td>40.00</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>358.1</td>
      <td>5.0</td>
      <td>87.6</td>
      <td>15.2</td>
      <td>26.3</td>
      <td>84.0</td>
      <td>65.31</td>
    </tr>
    <tr>
      <th>Oklahoma</th>
      <td>510.4</td>
      <td>6.5</td>
      <td>85.6</td>
      <td>16.2</td>
      <td>25.9</td>
      <td>75.4</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>261.2</td>
      <td>2.3</td>
      <td>89.1</td>
      <td>14.3</td>
      <td>22.7</td>
      <td>85.6</td>
      <td>62.47</td>
    </tr>
    <tr>
      <th>Pennsylvania</th>
      <td>388.9</td>
      <td>5.4</td>
      <td>87.9</td>
      <td>12.5</td>
      <td>24.5</td>
      <td>83.5</td>
      <td>70.68</td>
    </tr>
    <tr>
      <th>Rhode Island</th>
      <td>254.3</td>
      <td>3.0</td>
      <td>84.7</td>
      <td>11.5</td>
      <td>27.3</td>
      <td>82.6</td>
      <td>90.46</td>
    </tr>
    <tr>
      <th>South Carolina</th>
      <td>675.1</td>
      <td>6.7</td>
      <td>83.6</td>
      <td>17.1</td>
      <td>28.4</td>
      <td>67.6</td>
      <td>55.78</td>
    </tr>
    <tr>
      <th>South Dakota</th>
      <td>201.0</td>
      <td>3.6</td>
      <td>89.9</td>
      <td>14.2</td>
      <td>20.8</td>
      <td>86.3</td>
      <td>29.92</td>
    </tr>
    <tr>
      <th>Tennessee</th>
      <td>666.0</td>
      <td>7.4</td>
      <td>83.1</td>
      <td>17.1</td>
      <td>26.3</td>
      <td>79.1</td>
      <td>54.38</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>491.4</td>
      <td>5.4</td>
      <td>79.9</td>
      <td>17.2</td>
      <td>27.6</td>
      <td>73.8</td>
      <td>75.35</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>216.2</td>
      <td>1.4</td>
      <td>90.4</td>
      <td>11.5</td>
      <td>17.9</td>
      <td>89.3</td>
      <td>81.17</td>
    </tr>
    <tr>
      <th>Vermont</th>
      <td>135.1</td>
      <td>1.3</td>
      <td>91.0</td>
      <td>11.4</td>
      <td>21.3</td>
      <td>95.8</td>
      <td>17.38</td>
    </tr>
    <tr>
      <th>Virginia</th>
      <td>230.0</td>
      <td>4.7</td>
      <td>86.6</td>
      <td>10.5</td>
      <td>24.0</td>
      <td>70.4</td>
      <td>69.79</td>
    </tr>
    <tr>
      <th>Washington</th>
      <td>338.3</td>
      <td>2.8</td>
      <td>89.7</td>
      <td>12.3</td>
      <td>22.2</td>
      <td>80.2</td>
      <td>74.97</td>
    </tr>
    <tr>
      <th>West Virginia</th>
      <td>331.2</td>
      <td>4.9</td>
      <td>82.8</td>
      <td>17.7</td>
      <td>23.3</td>
      <td>94.3</td>
      <td>33.20</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <td>259.7</td>
      <td>2.6</td>
      <td>89.8</td>
      <td>12.4</td>
      <td>22.2</td>
      <td>88.4</td>
      <td>55.80</td>
    </tr>
    <tr>
      <th>Wyoming</th>
      <td>219.3</td>
      <td>2.0</td>
      <td>91.8</td>
      <td>9.8</td>
      <td>18.9</td>
      <td>91.3</td>
      <td>24.51</td>
    </tr>
  </tbody>
</table>
</div>



This `d` dataset object has more attributes (see details here: https://www.statsmodels.org/stable/datasets/index.html#available-datasets) amonst others they have been pre-partitioned into exogenous and endogenous variables.


```python
print(d.endog_name)
d.endog.head(n=10) # only showing first 10 rows
```

    murder
    




    state
    Alabama                  7.1
    Alaska                   3.2
    Arizona                  5.5
    Arkansas                 6.3
    California               5.4
    Colorado                 3.2
    Connecticut              3.0
    Delaware                 4.6
    District of Columbia    24.2
    Florida                  5.5
    Name: murder, dtype: float64




```python
print(d.exog_name)
d.exog.head(n=10)
```

    ['urban', 'poverty', 'hs_grad', 'single']
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>urban</th>
      <th>poverty</th>
      <th>hs_grad</th>
      <th>single</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama</th>
      <td>48.65</td>
      <td>17.5</td>
      <td>82.1</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>44.46</td>
      <td>9.0</td>
      <td>91.4</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <td>80.07</td>
      <td>16.5</td>
      <td>84.2</td>
      <td>25.7</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <td>39.54</td>
      <td>18.8</td>
      <td>82.4</td>
      <td>26.3</td>
    </tr>
    <tr>
      <th>California</th>
      <td>89.73</td>
      <td>14.2</td>
      <td>80.6</td>
      <td>27.8</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>76.86</td>
      <td>12.9</td>
      <td>89.3</td>
      <td>21.4</td>
    </tr>
    <tr>
      <th>Connecticut</th>
      <td>84.83</td>
      <td>9.4</td>
      <td>88.6</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>Delaware</th>
      <td>68.71</td>
      <td>10.8</td>
      <td>87.4</td>
      <td>27.6</td>
    </tr>
    <tr>
      <th>District of Columbia</th>
      <td>100.00</td>
      <td>18.4</td>
      <td>87.1</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>87.44</td>
      <td>14.9</td>
      <td>85.3</td>
      <td>26.6</td>
    </tr>
  </tbody>
</table>
</div>



Before we can estimate a regression model we specify the regression model and save that specification in an object `mod`. The first argument specifies the explained and the second the explanatory variables. The result is an object of the model class OLS (see the above help on statsmodels for other type of model classes). 


```python
mod = sm.OLS(d.endog,d.exog)
```

In order to confirm what type of object `mod` is you could run `type(mod)` which will confirm `mod` is of type "statsmodels.regression.linear_model.OLS". Every object of that type has some attributes and methods associated with it. You can figure out which by running `dir(mod)`. 

You can think of attributes as characteristics of the object and of methods as of tools that can be applied to the object. The method that is of immediate importance is to actually estimate (or `fit`) the model. We apply that method to the object `mod` using the command `mod.fit()`. The result we save in `res`.


```python
res = mod.fit()
```


```python
print(res.summary())
```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                 murder   R-squared (uncentered):                   0.915
    Model:                            OLS   Adj. R-squared (uncentered):              0.908
    Method:                 Least Squares   F-statistic:                              126.9
    Date:                Thu, 17 Aug 2023   Prob (F-statistic):                    1.45e-24
    Time:                        17:17:41   Log-Likelihood:                         -101.53
    No. Observations:                  51   AIC:                                      211.1
    Df Residuals:                      47   BIC:                                      218.8
    Df Model:                           4                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    urban         -0.0118      0.016     -0.745      0.460      -0.044       0.020
    poverty        0.0348      0.106      0.327      0.745      -0.179       0.249
    hs_grad       -0.1187      0.016     -7.614      0.000      -0.150      -0.087
    single         0.6137      0.079      7.803      0.000       0.455       0.772
    ==============================================================================
    Omnibus:                        3.288   Durbin-Watson:                   2.523
    Prob(Omnibus):                  0.193   Jarque-Bera (JB):                2.663
    Skew:                           0.200   Prob(JB):                        0.264
    Kurtosis:                       4.046   Cond. No.                         53.0
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

This is very much like a standard regression output you would see from most statistical computing packages. One thing you may note is that there are two degrees of freedom (Df) information. The model and residual degrees of freedom. The model Df tells you how many explanatory variables were used (here 4) and the residual Df is the number of observations minus the number of estimated coefficients, here 51 - 4 = 47. The latter is the usual definition of degrees of freedom in the context of regression models. 


```python

```
