# Recurrent Neural Networks

This page will iillustrate a simple implementation of a recurrent neural network (RNN). The example we will replicate is that of inflation forecasting discussed in [Almosova and Andresen, 2022, Journal of Forecasting](https://ideas.repec.org/a/wly/jforec/v42y2023i2p240-259.html). In this paper the authors use AR, seasonal AR, RNN and a long short-term memory recurrent neural network (LSTM) to forecast US CPI inflation.

## The data

The data used in the above paper is US Consumer Price Index inflation (CPALTT01USM657N). The data is available from the [St. Louis FED FRED Database](https://fred.stlouisfed.org/series/CPALTT01USM657N). Download the data into a csv file and save this as CPALTT01USM657N.csv in a folder called "datasets" under your main working directory.

## Preparing your code

In order to work with neural networks we want to employ a package which contains all the necessary functionality.



```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

Now load the data. If you were to check the downloaded csv file you will see that there are two columns of data: `DATE` and  `CPALTT01USM657N`. And by default, python will add an index column.


```python
# Load the data
data = pd.read_csv('datasets/CPALTT01USM657N.csv')
data.head()
```




    DATE                object
    CPALTT01USM657N    float64
    dtype: object



We want to change the `DATE` column into a recognised date format and we want this date info to be the index for the dataframe. We also want to change the name of the inflation series to `inflation`. This is achieved in the following:


```python
data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)
data.rename(columns={'CPALTT01USM657N': 'inflation'})
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
      <th>inflation</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960-01-01</th>
      <td>-0.340136</td>
    </tr>
    <tr>
      <th>1960-01-02</th>
      <td>0.341297</td>
    </tr>
    <tr>
      <th>1960-01-03</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1960-01-04</th>
      <td>0.340136</td>
    </tr>
    <tr>
      <th>1960-01-05</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-01-03</th>
      <td>0.331073</td>
    </tr>
    <tr>
      <th>2023-01-04</th>
      <td>0.505904</td>
    </tr>
    <tr>
      <th>2023-01-05</th>
      <td>0.251844</td>
    </tr>
    <tr>
      <th>2023-01-06</th>
      <td>0.322891</td>
    </tr>
    <tr>
      <th>2023-01-07</th>
      <td>0.190752</td>
    </tr>
  </tbody>
</table>
<p>763 rows × 1 columns</p>
</div>






```python

```
