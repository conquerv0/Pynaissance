# Asset Pricing with Factor Models,

**I. Capital Asset Pricing Model(CAPM)**

  Core Assumptions:
- Investors are risk-averse, utility maximizing, rational.
- Choices are based on expected risk, return and normality.
- One period market window.
- Homogenous outlook, no private information.
- Borrow and lend at risk-free rate.
- Market are frictionless and investment can be infinitely divisible. 

In equilibrium, the best portfolio must be the market portfolio that every participants tends to hold.
Such, a Security Market Line formed to represent how individual securities are priced. SML represents such tradeoff between Expected return and Market Risk(beta).

![SML Equation](https://latex.codecogs.com/gif.latex?E%28R_i%29%3DR_f&plus;%5Cbeta_i%28R_m-R_f%29)
![CAPM SML](https://cdn.wallstreetmojo.com/wp-content/uploads/2018/08/security-market-line.jpg)

注意，仅有系统性市场风险beta在资产组合中被定价了。这与基本假设相呼应，因为一切非系统性风险都可以通过分散性投资分化和消除，所以不应被”定价“。

  Core Remarks:
- All individuals hold a portfolio along the CML, with combination of Rf and M.
- All firm-specific risk(non-systematic) risk can be diversifed away and thus not priced.
- All invidual securities are priced based purely on its market risk(as measured in beta/covariance with market), not the total volatility risk. 

基本的CAPM定价模型对任何“非市场择时者”，“选股者”来说是最完美的模型，个人投资者只需要依照自己的风险偏好选取一个风险合理分化的资产组合。CAPM严格意义上来说就是一个以market premium，市场风险溢价为因子的单因子模型。

**II. Portfolio Measures: Beta and Alpha**

**III. Fundamental: Factor Basics**

- Value: mostly track core ratios in fundamental analysis, aims to capture excess returns from stocks that have low prices relative to their fundamental value. Commonly tracked by price to book, price to earnings, dividends, and free cash flow. 
  
- Size: Market capitalization in some way levy upper limits on the long term growth of the stock. Small-cap stocks that survived and prospered outperformed those of lower risk large cap stocks. 

- Momentum: As stocks that either underperformed or outperformed the market tend to have some market lags, such past performance exhibits strength or weakness going forward. A momentum strategy can look at the respective market window on weekly, monthly or even yearly basis.

- Quality: Another factor that tracks core ratio in fundamental analysis. As quality is defined by low debt, stable earnings, consistent asset growth, and strong corporate governance. Common financial metrics such as return to equity, debt to equity and earnings variability can be used to track quality factor. 

**IV. Factor Model**

Rates of return are related to factors.

Single-factor Model：

![SML Equation](https://latex.codecogs.com/svg.latex?R_i=\alpha_i+\beta_if+e_i)

Where：

![SML Equation](https://latex.codecogs.com/svg.latex?Ri) is the rates of return on i asset

![SML Equation](https://latex.codecogs.com/svg.latex?\alpha_i) and ![SML Equation](https://latex.codecogs.com/svg.latex?\beta_i) are constant, ![SML Equation](https://latex.codecogs.com/svg.latex?\beta_i) is the factor loading

![SML Equation](https://latex.codecogs.com/svg.latex?f) is a random factor in one factor model

![SML Equation](https://latex.codecogs.com/svg.latex?e_i) is a random error with ![SML Equation](https://latex.codecogs.com/svg.latex?E[e_i]=0)

单因子模型中，证券的超额收益只受单一因素的影响，例如CAPM。而因子需要是所有证券的共同因子，例如CAPM中的因子为market premium。

Multi-factor Model:

![SML Equation](https://latex.codecogs.com/svg.latex?R_i=\alpha_i+\sum_j^n\beta_i_jf_j+e_i)

同理，在多因子模型中，证券的超额收益受多个因子影响，但不同因子直接需满足条件：

- Factors are uncorrelated

- Error terms on any two asset are uncorrelated

- Error term is uncorrelated with any factor

Possible factors:

- External Factors: GDP, PPI, CPI, ...

- Extracted Factors: Market Portfolio, Industry Averages, ...


**V. Fama & French Three-Factor Model (1993, 1996)**

![SML Equation](https://latex.codecogs.com/svg.latex?R_i_t-R_f_t=\alpha_i+\beta_i_M(R_M_t-R_f_t)+\beta_i_sSMB_t+\beta_i_hHML_t+e_i_t)

Where：

SMB is "Small Minus Big" size risk 市值因子

HML is "High Minus Low" value risk 账面市值比（B/M）因子

**VI. Carhart Four-Factor Model (1997)**

![SML Equation](https://latex.codecogs.com/svg.latex?R_i_t-R_f_t=\alpha_i+\beta_i_M(R_M_t-R_f_t)+\beta_i_sSMB_t+\beta_i_hHML_t+\beta_i_uUMD_t+e_i_t)

Where:

UMD is "Up Minus Down" momentum risk 动量因子

**VII. Fama & French Five-Factor Model (2015,2016)**

![SML Equation](https://latex.codecogs.com/svg.latex?R_i_t-R_f_t=\alpha_i+\beta_i_M(R_M_t-R_f_t)+\beta_i_sSMB_t+\beta_i_hHML_t+\beta_i_rRMW_t+\beta_i_cCMA_t+e_i_t)

Where:

RMW is "Robust Minus Weak" operating profitablity risk 盈利水平因子

CMA is "Conservative Minus Aggressive" investment risk 投资水平因子
