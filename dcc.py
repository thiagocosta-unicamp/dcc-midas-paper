import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.figure_factory as ff
from arch import arch_model
from ipywidgets import HBox, VBox, Dropdown, Output
from scipy.optimize import fmin, minimize
from scipy.stats import t
from scipy.stats import norm
from math import inf
from IPython.display import display
import yfinance as yf
import datetime
import streamlit as st

#%% Dynamic Conditional Correlation

def vecl(matrix):
    """
    Extracts the elements below the main diagonal of a matrix, removing zero.

    This function is useful in the context of DCC-GARCH models where the aim is to capture only the correlations between pairs of variables, excluding the main diagonal (which typically contains variances or unit values in the case of correlation matrices).

    Parameters:
    matrix (ndarray): A square matrix from which the elements below the main diagonal are to be extracted.

    Returns:
    ndarray: A one-dimensional array containing the elements below the main diagonal.
    
    """
    # Extracts the lower triangular part of the matrix, excluding the main diagonal.
    # The argument k=-1 indicates that the main diagonal (k=0) and all elements above it will not be included.
    lower_matrix = np.tril(matrix,k=-1)

    # Converts the lower triangular matrix to a 1D array.
    array_with_zero = np.matrix(lower_matrix).A1

    # Removes zero values from the array.
    array_without_zero = array_with_zero[array_with_zero!=0]
   
    return array_without_zero

def garch_t_to_u(rets, res):
    """
    Transforms returns adjusted by a GARCH model into uniformly distributed data.

    This function performs the following steps:
    1. Calculates the adjusted returns by subtracting the estimated mean (mu) from the observed returns.
    2. Divides the adjusted returns by the estimated conditional volatilities to obtain the standardized residuals.
    3. Uses the cumulative distribution function (CDF) of the t-distribution to transform the standardized residuals into uniformly distributed data.

    Parameters:
    rets (ndarray): Array of returns.
    res (GARCHResults): Result of the GARCH model fitting.

    Returns:
    udata (ndarray): Uniformly distributed data resulting from the transformation.
    """

    # Extract the mean (mu) from the GARCH model results.
    mu = res.params['mu']
    
    # Extract the degrees of freedom (nu) from the t-distribution from the GARCH model results.
    nu = res.params['nu']
    
    # Subtract the mean from the returns to get the adjusted returns.
    est_r = rets - mu

    # Extract the conditional volatility from the GARCH model results.
    h = res.conditional_volatility
   
    # Divide the adjusted returns by the conditional volatility to get the standardized residuals.
    std_res = est_r / h
    # we could also just use:
    # std_res = res.std_resid
    # but it's useful to see what is going on
    
    # Transform the standardized residuals to uniformly distributed data using the CDF of the t-distribution.
    udata = t.cdf(std_res, nu)
    
    # Retornamos os dados uniformemente distribuídos.
    return udata

def loglike_norm_dcc_copula(theta, udata):
    """
    Calculates the log-likelihood of a DCC (Dynamic Conditional Correlation) normal copula
    for a set of uniformly distributed data.

    Parameters:
    - theta: Parameters of the DCC model to be optimized.
    - udata: Uniformly distributed data, resulting from the transformation of standardized
             residuals from a GARCH model fitted to each time series.

    Returns:
    - The negative value of the log-likelihood for optimization (minimization) purposes.
    """
    
    # Obtain the number of time series (N) and the number of time observations (T)
    N, T = np.shape(udata)

    # Initialize the vector to store the log-likelihood values over time
    llf = np.zeros((T,1))
    
    # Transform the uniform data to the normal space using the inverse quantile function
    trdata = np.array(norm.ppf(udata).T, ndmin=2)
    
    # Calculate the conditional dynamic correlation matrices (Rt) and the column vectors of the conditional dynamic correlation matrices (veclRt)
    Rt, veclRt =  dcceq(theta,trdata)

    # Loop through each time point T to calculate the log-likelihood
    for i in range(0,T):
        # Calculate the normalization term of the log-likelihood considering the determinant of the dynamic correlation matrix
        llf[i] = -0.5* np.log(np.linalg.det(Rt[:,:,i]))

        # Calculate the term involving the inverse of the dynamic correlation matrix, adjusted by the transformed data
        llf[i] = llf[i] - 0.5 *  np.matmul(np.matmul(trdata[i,:] , (np.linalg.inv(Rt[:,:,i]) - np.eye(N))) ,trdata[i,:].T)
    
    # Sum the log-likelihood values over all time points
    llf = np.sum(llf)

    # Return the negative value of the log-likelihood for optimization (minimization) purposes
    return -llf

def dcceq(theta,trdata):
    T, N = np.shape(trdata)

    a, b = theta
    
    # Ensure that the parameters a and b are within acceptable limits
    if min(a,b)<0 or max(a,b)>1 or a+b > .999999:
        a = .9999 - b

    # Initialize the matrices 
    Qt = np.zeros((N, N ,T)) # Dynamic covariance matrix
    Qt[:,:,0] = np.cov(trdata.T) # Initial covariance (period 0)

    Rt =  np.zeros((N, N ,T))  # Dynamic correlation matrix
    veclRt =  np.zeros((T, int(N*(N-1)/2))) # Vector to store correlations

    # Initial correlation matrix
    Rt[:,:,0] = np.corrcoef(trdata.T) # Initial correlation matrix
    
    # Loop to calculate the dynamic covariance and correlation matrices
    for j in range(1,T):
        # Calculate the dynamic covariance matrix
        Qt[:,:,j] = Qt[:,:,0] * (1-a-b)
        Qt[:,:,j] = Qt[:,:,j] + a * np.matmul(trdata[[j-1]].T, trdata[[j-1]])
        Qt[:,:,j] = Qt[:,:,j] + b * Qt[:,:,j-1]
        
        # Calculate the dynamic correlation matrix
        # Normalize the covariance matrix to obtain the correlation matrix
        Rt[:,:,j] = np.divide(Qt[:,:,j] , np.matmul(np.sqrt(np.array(np.diag(Qt[:,:,j]), ndmin=2)).T , np.sqrt(np.array(np.diag(Qt[:,:,j]), ndmin=2))))
    
    # Extract the correlations in vector format
    for j in range(0,T):
        veclRt[j, :] = vecl(Rt[:,:,j].T)
    return Rt, veclRt

def run_garch_on_return(rets, udata_list, model_parameters):
    # For each series of returns 'x' in the 'rets' dictionary
    for x in rets:
        # Fit a GARCH model with t-distribution to the return series
        am = arch_model(rets[x], dist = 't')
        
        # Extract a short name for the series, removing spaces
        short_name = x.split()[0]

        # Fit the model and store the results in the model parameters
        model_parameters[short_name] = am.fit(disp='off')

        # Convert the fitted returns to uniform data using the garch_t_to_u function
        udata = garch_t_to_u(rets[x], model_parameters[short_name])
        
        # Add the uniform data to the list of uniform data
        udata_list.append(udata)

    # Return the list of uniform data and the fitted model parameters
    return udata_list, model_parameters

#%%Initially run GARCH on the individualtime series, and transform them to the uniform distribution

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2020,12,30)

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ']

close_prices = yf.download(tickers, start=start, end=end)['Adj Close']

rets = ((close_prices / close_prices.shift(1)) - 1 ).dropna(how='all') * 100
rets = rets.loc[:, rets.var() != 0]  # Remove columns with zero variance
rets = rets.dropna(how='any')        # Remove rows with NaN values

model_parameters = {}
udata_list = []

udata_list, model_parameters = run_garch_on_return(rets, udata_list, model_parameters)

#%% Setup our DDC Model, and then run it on our 5 securities

cons = ({'type': 'ineq', 'fun': lambda x:  -x[0]  -x[1] +1})
bnds = ((0, 0.5), (0, 0.9997))

opt_out = minimize(loglike_norm_dcc_copula, [0.01, 0.95], args = (udata_list,), bounds=bnds, constraints=cons)

#print(opt_out.success)
#print(opt_out.x)

llf  = loglike_norm_dcc_copula(opt_out.x, udata_list)
#llf

trdata = np.array(norm.ppf(udata_list).T, ndmin=2)
Rt, veclRt = dcceq(opt_out.x, trdata)

stock_names = [x.split()[0] for x in rets.columns]

corr_name_list = []
for i, name_a in enumerate(stock_names):
    if i == 0:
        pass
    else:
        for name_b in stock_names[:i]:
            corr_name_list.append(name_a + "-" + name_b)

#%% Graphs

dcc_corr = pd.DataFrame(veclRt, index = rets.index, columns= corr_name_list)
dcc_plot = px.line(dcc_corr, title = 'Dynamic Conditional Correlation plot', width=1000, height=500)
#dcc_plot.show()

garch_vol_df = pd.concat([pd.DataFrame(model_parameters[x].conditional_volatility/100)*1600 for x in model_parameters], axis=1)
garch_vol_df.columns = stock_names

px.line(garch_vol_df, title='GARCH Conditional Volatility', width=1000, height=500)

#px.scatter(garch_vol_df, x = 'AAPL', y='AMZN', width=1000, height=500, title='GARCH Volatility')

px.line(np.log((1+rets/100).cumprod()), title='Cumulative Returns', width=1000, height=500)

#rets.loc[:, ['ABBV','A']].corr()

def update_corr_data(change):
    a1corr = rets.loc[:, pair_dropdown.value.split('-')].corr().values[0][1]
    a1dcc = pd.DataFrame(veclRt[:,corr_name_list.index(pair_dropdown.value)],index = rets.index)
    a1dcc.columns = ['DCC']
    a1dcc['corr'] = a1corr
    corr_line_plot = px.line(a1dcc, title = 'DCC vs unconditional correlation for ' + pair_dropdown.value, width=1000, height=500)
    output_graphics.clear_output()
    with output_graphics:
        display(corr_line_plot)

output_graphics = Output()
pair_dropdown = Dropdown(options=[''] + corr_name_list)
pair_dropdown.observe(update_corr_data, 'value')
VBox([pair_dropdown, output_graphics])     

#%%Streamlit app

# Set streamlit page configuration
st.set_page_config(page_title="Results", layout="wide")

# Greeting
st.title('Hi, Christian!')

# Streamlit UI
st.title('DCC-GARCH Model Results')

st.header('Overview and Methodology')
st.markdown("""
This application demonstrates the implementation and results of a Dynamic Conditional Correlation (DCC) model combined with GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models for volatility modeling. The process involves several steps:

### Step 1: Data Collection
We collect historical stock price data for multiple stocks using the `yfinance` library. The returns are calculated as the percentage change in adjusted closing prices.

### Step 2: Fitting GARCH Models
For each stock's return series, we fit a GARCH model with t-distribution. The fitted models provide the conditional volatilities and standardized residuals.

#### GARCH Model:
The GARCH(1,1) model can be represented as:
""")

st.latex(r'''
r_t = \mu + \epsilon_t
''')

st.latex(r'''
\epsilon_t = \sigma_t z_t
''')

st.latex(r'''
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
''')

st.markdown(r"""
where $r_t$ is the return at time $t$, $\sigma_t$ is the conditional volatility, and $z_t$ is the error term which follows a t-distribution.

### Step 3: Transforming to Uniform Data
The standardized residuals from the GARCH models are transformed to uniformly distributed data using the cumulative distribution function (CDF) of the t-distribution.

### Step 4: Estimating the DCC Model
The DCC model is estimated by maximizing the log-likelihood of a normal copula constructed from the transformed uniform data.

#### DCC Model:
The DCC model can be represented as:
""")

st.latex(r'''
Q_t = (1 - a - b) \bar{Q} + a u_{t-1} u_{t-1}' + b Q_{t-1}
''')

st.latex(r'''
R_t = Q_t^* Q_t^{-1} Q_t^*
''')

st.markdown(r"""
where $Q_t$ is the dynamic covariance matrix, $\bar{Q}$ is the unconditional covariance matrix of the standardized residuals, and $R_t$ is the dynamic correlation matrix.

### Step 5: Visualizing the Results
The dynamic correlations and conditional volatilities are plotted using `Plotly`.
""")

st.header('Dynamic Conditional Correlation (Use double-click to isolate one variable)')
st.subheader('Double-click on the legend to isolate one variable')
dcc_plot = px.line(dcc_corr, title='Dynamic Conditional Correlation plot', width=1000, height=500)
st.plotly_chart(dcc_plot)

st.header('GARCH Conditional Volatility')
garch_vol_plot = px.line(garch_vol_df, title='GARCH Conditional Volatility', width=1000, height=500)
st.plotly_chart(garch_vol_plot)

st.header('Cumulative Returns')
cumulative_returns = np.log((1 + rets / 100).cumprod())
cumulative_returns_plot = px.line(cumulative_returns, title='Cumulative Returns', width=1000, height=500)
st.plotly_chart(cumulative_returns_plot)

st.header('DCC vs Unconditional Correlation')
selected_pair = st.selectbox('Select Pair', [''] + corr_name_list)
if selected_pair:
    a1corr = rets.loc[:, selected_pair.split('-')].corr().values[0][1]
    a1dcc = pd.DataFrame(veclRt[:, corr_name_list.index(selected_pair)], index=rets.index)
    a1dcc.columns = ['DCC']
    a1dcc['Unconditional Corr'] = a1corr
    corr_line_plot = px.line(a1dcc, title=f'DCC vs Unconditional Correlation for {selected_pair}', width=1000, height=500)
    st.plotly_chart(corr_line_plot)
