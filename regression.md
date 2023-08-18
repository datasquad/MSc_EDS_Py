# Regression

Here we perform basic regresison analysis


## Using statsmodels

This follows the example in Kevin Sheppard's [Introduction to Python](https://www.kevinsheppard.com/files/teaching/python/notes/python_introduction_2021.pdf) Chapter 21.1 Regression. The statsmodels package has good documentation [here](https://www.statsmodels.org/stable/index.html).




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



This `d` dataset object has more attributes (see details [here](https://www.statsmodels.org/stable/datasets/index.html#available-datasets)) amonst others they have been pre-partitioned into exogenous and endogenous variables.


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
    Time:                        20:55:50   Log-Likelihood:                         -101.53
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

What you will notice here is that the regression did not include a constant. That is because our matrix with explanatory variabbles (`d.exog`) did not contain any. Often statistical procedures will automatically include a constant (like the `lm` function in R), but this one does not. So we need to actively add a constant. This is done with the add_constant function in statsmodels: `sm.add_constant(d.exog)`. So let's re-specify and reestimate the model with a constant.


```python
mod = sm.OLS(d.endog,sm.add_constant(d.exog))
res = mod.fit()
print(res.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 murder   R-squared:                       0.813
    Model:                            OLS   Adj. R-squared:                  0.797
    Method:                 Least Squares   F-statistic:                     50.08
    Date:                Thu, 17 Aug 2023   Prob (F-statistic):           3.42e-16
    Time:                        21:11:25   Log-Likelihood:                -95.050
    No. Observations:                  51   AIC:                             200.1
    Df Residuals:                      46   BIC:                             209.8
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -44.1024     12.086     -3.649      0.001     -68.430     -19.774
    urban          0.0109      0.015      0.707      0.483      -0.020       0.042
    poverty        0.4121      0.140      2.939      0.005       0.130       0.694
    hs_grad        0.3059      0.117      2.611      0.012       0.070       0.542
    single         0.6374      0.070      9.065      0.000       0.496       0.779
    ==============================================================================
    Omnibus:                        1.618   Durbin-Watson:                   2.507
    Prob(Omnibus):                  0.445   Jarque-Bera (JB):                0.831
    Skew:                          -0.220   Prob(JB):                        0.660
    Kurtosis:                       3.445   Cond. No.                     5.80e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.8e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

The `res` object we used to store the regression results in can be thought of as a shelf full of interesting information. The `summary()` method gave as the big hits from that shelf of of information. But you can access all possible individual elemenst from that shelf. To find out what is on that shelf you can again use the `dir(res)` command.


```python
dir(res)
```




    ['HC0_se',
     'HC1_se',
     'HC2_se',
     'HC3_se',
     '_HCCM',
     '__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     '_abat_diagonal',
     '_cache',
     '_data_attr',
     '_data_in_cache',
     '_get_robustcov_results',
     '_is_nested',
     '_use_t',
     '_wexog_singular_values',
     'aic',
     'bic',
     'bse',
     'centered_tss',
     'compare_f_test',
     'compare_lm_test',
     'compare_lr_test',
     'condition_number',
     'conf_int',
     'conf_int_el',
     'cov_HC0',
     'cov_HC1',
     'cov_HC2',
     'cov_HC3',
     'cov_kwds',
     'cov_params',
     'cov_type',
     'df_model',
     'df_resid',
     'diagn',
     'eigenvals',
     'el_test',
     'ess',
     'f_pvalue',
     'f_test',
     'fittedvalues',
     'fvalue',
     'get_influence',
     'get_prediction',
     'get_robustcov_results',
     'info_criteria',
     'initialize',
     'k_constant',
     'llf',
     'load',
     'model',
     'mse_model',
     'mse_resid',
     'mse_total',
     'nobs',
     'normalized_cov_params',
     'outlier_test',
     'params',
     'predict',
     'pvalues',
     'remove_data',
     'resid',
     'resid_pearson',
     'rsquared',
     'rsquared_adj',
     'save',
     'scale',
     'ssr',
     'summary',
     'summary2',
     't_test',
     't_test_pairwise',
     'tvalues',
     'uncentered_tss',
     'use_t',
     'wald_test',
     'wald_test_terms',
     'wresid']



Here are a few examples of what you could extract from `res`.


```python
print("\nConfidence intervals for coefficient estimates")
print(res.conf_int())

print("\nAIC information criterion")
print(res.aic)

print("\nRegression fitted values")
print(res.fittedvalues)
```

    
    Confidence intervals for coefficient estimates
                     0          1
    const   -68.430362 -19.774469
    urban    -0.020104   0.041880
    poverty   0.129901   0.694399
    hs_grad   0.070059   0.541795
    single    0.495840   0.778910
    
    AIC information criterion
    200.10018977656853
    
    Regression fitted values
    state
    Alabama                  7.240371
    Alaska                   4.305784
    Arizona                  5.709436
    Arkansas                 6.047842
    California               5.103820
    Colorado                 3.010261
    Connecticut              3.734915
    Delaware                 5.426470
    District of Columbia    21.810162
    Florida                  6.040398
    Georgia                  7.752260
    Hawaii                   5.380746
    Idaho                    1.495336
    Illinois                 5.253719
    Indiana                  4.585734
    Iowa                     1.839635
    Kansas                   3.940428
    Kentucky                 5.193414
    Louisiana                8.856294
    Maine                    2.869247
    Maryland                 4.940706
    Massachusetts            4.287778
    Michigan                 6.504815
    Minnesota                1.930016
    Mississippi             10.726801
    Missouri                 5.211376
    Montana                  3.126335
    Nebraska                 2.345950
    Nevada                   5.782611
    New Hampshire            0.276226
    New Jersey               3.958383
    New Mexico               7.779862
    New York                 7.810840
    North Carolina           5.765752
    North Dakota             0.319488
    Ohio                     6.435508
    Oklahoma                 5.768319
    Oregon                   4.197993
    Pennsylvania             4.325676
    Rhode Island             4.934577
    South Carolina           7.229609
    South Dakota             2.836099
    Tennessee                5.722915
    Texas                    5.842075
    Utah                     0.585888
    Vermont                  2.200749
    Virginia                 2.775294
    Washington               3.374663
    West Virginia            3.735693
    Wisconsin                3.237746
    Wyoming                  0.333984
    dtype: float64
    

## Robust standard errors

The info in the previous regression output highlights that "Covariance Type: nonrobust". This means that the coefficient standard errors have been calculated with the standard formula which assumes that error terms are iid distributed (also see the warning note [1] in the regression summary output). You will have learned that it is almost standard practice to calculate heteroskedasticity-robust (or heteroskedasticity and autocorrelation-robust) standard errors. Often they are referred to as White (Newey-West standard errors). If you want these you will have to let the `.fit` method know. 


```python
res_white=mod.fit(cov_type='HC0')
print(res_white.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 murder   R-squared:                       0.813
    Model:                            OLS   Adj. R-squared:                  0.797
    Method:                 Least Squares   F-statistic:                     31.45
    Date:                Thu, 17 Aug 2023   Prob (F-statistic):           1.23e-12
    Time:                        21:21:19   Log-Likelihood:                -95.050
    No. Observations:                  51   AIC:                             200.1
    Df Residuals:                      46   BIC:                             209.8
    Df Model:                           4                                         
    Covariance Type:                  HC0                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -44.1024     11.873     -3.715      0.000     -67.373     -20.832
    urban          0.0109      0.013      0.814      0.416      -0.015       0.037
    poverty        0.4121      0.115      3.595      0.000       0.187       0.637
    hs_grad        0.3059      0.111      2.763      0.006       0.089       0.523
    single         0.6374      0.082      7.733      0.000       0.476       0.799
    ==============================================================================
    Omnibus:                        1.618   Durbin-Watson:                   2.507
    Prob(Omnibus):                  0.445   Jarque-Bera (JB):                0.831
    Skew:                          -0.220   Prob(JB):                        0.660
    Kurtosis:                       3.445   Cond. No.                     5.80e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors are heteroscedasticity robust (HC0)
    [2] The condition number is large, 5.8e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

The regression output illustrates that this did not change the coefficient estimates but only the standard errors of the coefficient estimates changed. They are now White standard errors.

If you wanted to implement Newey-West standard errors (here using 2 lags) to make standard errors robust to the presence of autocorrelation you would use the following. 


```python
res_NW=mod.fit(cov_type='HAC',cov_kwds={'maxlags':2})
print(res_NW.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 murder   R-squared:                       0.813
    Model:                            OLS   Adj. R-squared:                  0.797
    Method:                 Least Squares   F-statistic:                     36.41
    Date:                Thu, 17 Aug 2023   Prob (F-statistic):           1.03e-13
    Time:                        21:51:20   Log-Likelihood:                -95.050
    No. Observations:                  51   AIC:                             200.1
    Df Residuals:                      46   BIC:                             209.8
    Df Model:                           4                                         
    Covariance Type:                  HAC                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -44.1024     12.488     -3.532      0.000     -68.578     -19.627
    urban          0.0109      0.014      0.804      0.422      -0.016       0.037
    poverty        0.4121      0.114      3.604      0.000       0.188       0.636
    hs_grad        0.3059      0.116      2.634      0.008       0.078       0.534
    single         0.6374      0.080      7.922      0.000       0.480       0.795
    ==============================================================================
    Omnibus:                        1.618   Durbin-Watson:                   2.507
    Prob(Omnibus):                  0.445   Jarque-Bera (JB):                0.831
    Skew:                          -0.220   Prob(JB):                        0.660
    Kurtosis:                       3.445   Cond. No.                     5.80e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors are heteroscedasticity and autocorrelation robust (HAC) using 2 lags and without small sample correction
    [2] The condition number is large, 5.8e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

Again the parameter estimates remain unchanged and the standard errors do change. 

This particular application of Newey-West standard erros, of course, makes no sense as these are no time-series data. This is therefore just another example that not everything you can do in your statistical software actually makes sense. You need to keep your thinking hat on all the time.
