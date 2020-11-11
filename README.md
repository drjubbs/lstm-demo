# LSTM Demo

Demonstration of using LSTM networks to forecast time series data. The demo includes four data sets from FRED, electricity and gas demand, and the 1, 5, and 10 year US T-bills yields. The objective is to forecast electricity demand using electricity consumption in an autoregressive capacity, and using the T-bill yields as additional predictors. The rationale for including these variables is that the shape of the yield curve is thought to correlate with economic health.

At the moment, the project only includes a single notebook which does the data normalization. The `jupytext` plugin is used to convert the Jupyter notebooks to markdown which works well with `diff`.

# Sample Data Sets

| *Filename*       | *Description*                                           |
| ---------------- | ------------------------------------------------------- |
| `DGS1.csv`       | FRED, 1-Year Treasury Constant Maturity Rate (DGS1)<br>Percent, Not Seasonally Adjusted, Updated: Nov 10, 2020                   |
| `DGS5.csv`       | FRED, 5-Year Treasury Constant Maturity Rate (DGS5)<br>Percent, Not Seasonally Adjusted, Updated: Nov 10, 2020                   |
| `DGS10.csv`      | FRED, 10-Year Treasury Constant Maturity Rate (DGS10)<br>Percent, Not Seasonally Adjusted, Updated: Nov 10, 2020                 |
| `IPG2211A2N.csv` | FRED, Industrial Production: Electric and Gas (NAICS = 2211,2)<br>Index 2012=100, Not Seasonally Adjusted, Updated: Oct 16, 2020 |
