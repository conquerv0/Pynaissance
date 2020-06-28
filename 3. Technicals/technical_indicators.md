# Techinical Analysis, A Foundation. 

1. **MACD ( Moving Average Convergence and Divergence)**

   Proposed by Geral Appel in 1979, this indicator aimes to utilize the convergence and divergence of the short term moving 
   average of the index clossing price(usually 12 days) and the long term moving average of the index closing price, to 
   assit market timing decision.
  
   MACD Formula Components:
   
  - Short term EMA(Exponnential moving average, a type of moving average that assigns greater weights to most recent data point)
  - Long term EMA: Long term exponential moving average of index price. 
  - DIF: Difference between short term EMA and long term EMA. 
  - DEA: Difference Exponential Average, The moving average of the DIF line in the given market window.
  - MACD: The difference between DIF and DEA.
  

2. **Bollinger Band**

   Developed by John Bollinger, this techinical tool is defined by a set of trendlines plotted two standard deviations(positively and negatively) 
   away from a simple moving average(SMA) of a security's price. 
   
   Bollinger Formula and Components:
   
   - BOLU: MA(TP, n) + m * sd[TP, n], Upper bollinger band.
   - BOLD: MA(TP, n) - m * sd[TP, n], Lower bollinger band.
   - MA: Moving average.
   - TP(Typical price) = (High + Low + Close) / 3
   - n: Number of days in smoothing period(typically 20)
   - m: Number of standard deviations
   - sd[TP, n]: Standard deviation over last n periods of TP. 
   
  
