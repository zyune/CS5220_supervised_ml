The model cleverly combines the classic Exponential Smoothing model (ES) and a Recurrent Neural Network (RNN)

- The ES decomposes the time series in level, trend and seasonality components.
- The RNN is trained with all the series, has shared parameters and it is used to learn common local trends among the series while the ES parameters are specific for each time series.

## what is forecasting?
