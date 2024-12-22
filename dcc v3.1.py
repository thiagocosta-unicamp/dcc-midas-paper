import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.figure_factory as ff
from arch import arch_model  # Importing GARCH model implementation from arch package
from ipywidgets import HBox, VBox, Dropdown, Output
from scipy.optimize import fmin, minimize  # For optimization (minimization)
from scipy.stats import t, norm  # For statistical distributions
from math import inf  # For infinity value
from IPython.display import display  # For displaying outputs in Jupyter
import yfinance as yf  # For downloading stock market data from Yahoo Finance
import datetime  # For handling date and time
import streamlit as st  # For creating web applications (interactive dashboards)
import matplotlib.pyplot as plt  # For plotting graphs
import pandas_datareader.data as web  # For downloading economic data (like CPI) from FRED
import numdifftools as ndt 

#%%Functions

def garch_t_to_u(rets, res):
    """
    This function transforms returns adjusted by a GARCH model into uniformly distributed data.

    Steps performed:
    1. Subtracts the estimated mean (mu) from observed returns to get adjusted returns.
    2. Divides the adjusted returns by the estimated conditional volatility to standardize them.
    3. Uses the CDF of the t-distribution to convert standardized residuals into uniformly distributed data.

    Args:
    rets (ndarray): Array of returns.
    res (GARCHResults): Result from fitting the GARCH model.

    Returns:
    udata (ndarray): Uniformly distributed data after transformation.
    """
    # Extract the mean (mu) from the GARCH model results
    mu = res.params['mu']
    
    # Extract the degrees of freedom (nu) for the t-distribution from the GARCH results
    nu = res.params['nu']
    
    # Adjust the returns by subtracting the mean (mu)
    est_r = rets - mu

    # Extract the conditional volatility (h) from the GARCH results
    h = res.conditional_volatility
   
    # Standardize the returns by dividing by the conditional volatility
    std_res = est_r / h
    
    # Transform the standardized residuals into uniformly distributed data using the CDF of the t-distribution
    udata = t.cdf(std_res, nu)
    
    return udata


def run_garch_on_return(rets, udata_list, model_parameters):
    """
    Fit a GARCH model to each series of returns in the 'rets' dictionary.

    Args:
    rets (dict): Dictionary of return series (e.g., S&P500, Bonds).
    udata_list (list): List to store the uniformly distributed data.
    model_parameters (dict): Dictionary to store the results of the fitted GARCH models.

    Returns:
    udata_list (list): List containing the uniformly distributed data from each series.
    model_parameters (dict): Dictionary containing the fitted GARCH model results.
    """
    for x in rets:
        # Fit a GARCH(1,1) model with t-distribution to the return series
        am = arch_model(rets[x], vol='GARCH', p=1, q=1, dist='t')
        
        # Shorten the series name for the key in the dictionary
        short_name = x.split()[0]

        # Fit the model and store the result in model_parameters
        model_parameters[short_name] = am.fit(disp='off')

        # Convert the fitted returns to uniform data using the garch_t_to_u function
        udata = garch_t_to_u(rets[x], model_parameters[short_name])
        
        # Append the uniform data to the list
        udata_list.append(udata)

    # Return the updated list of uniform data and model parameters
    return udata_list, model_parameters


def plot_conditional_volatility(rets, model_parameters):
    """
    Plot the conditional volatility of the GARCH model for each return series.

    Args:
    rets (dict): Dictionary of return series.
    model_parameters (dict): Dictionary containing the fitted GARCH models.
    """
    for x in rets:
        # Extract the short name for the series
        short_name = x.split()[0]
        
        # Retrieve the results of the fitted model for that series
        res = model_parameters[short_name]
        
        # Extract the conditional volatility from the model results
        volatility = res.conditional_volatility
                  
        # Plot the volatility
        plt.figure(figsize=(12, 6))
        plt.plot(volatility, label='Conditional Volatility', color='blue')
        plt.title(f'Conditional Volatility for {short_name}')
        plt.xlabel('Date')
        plt.ylabel('Conditional Volatility (%)')
        plt.legend()
        plt.grid()
        plt.show()


def loglikelihood_garch(params, udata_list, K, lambda_param, MV = None, RC = None):
    """
    Calculates the log-likelihood for a GARCH(1,1) model with optional macroeconomic variables (`MV`) and realized correlations (`RC`).

    Args:
        params (ndarray): Model parameters.
        udata_list (list): List of standardized residuals.
        K (int): Number of lags for macroeconomic variables.
        lambda_param (float): Decay parameter for exponential weights.
        MV (ndarray, optional): Macro variables.
        RC (ndarray, optional): Realized correlations.

    Returns:
        float: Negative log-likelihood value.
    """

    if MV is None and RC is None:
        print("Error: Both MV and RC are missing. Please provide at least one of them to proceed.")
        raise ValueError("Both MV and RC are missing. Provide at least one of these inputs.")

    if MV is not None and RC is not None:
        alpha, beta, theta1, theta_rc = params

    if MV is not None and RC is None:
        alpha, beta, theta1 = params
    
    if MV is None and RC is not None:
        alpha, beta, theta_rc = params
    
    #Apagar depois
    # alpha = 0.05
    # beta = 0.9
    # theta1 = 0.1
    # theta_rc = 0.1
    # b = 'Yes'
    # X = "Yes"

    T = len(udata_list[0])
    q_11 = np.zeros(T)
    q_12 = np.zeros(T)
    q_22 = np.zeros(T)
    lt_rho = np.zeros(T)
    m1 = np.zeros(T)
    m_rc = np.zeros(T)
    rho = np.zeros(T)
    
    z1 = udata_list[0]  # Residuals of S&P 500
    z2 = udata_list[1]  # Residuals of US Bond

    # Initialize weights for lagged realized correlations
    phi = np.zeros(K + 1)
    for i in range(1, K + 1):
        phi[i] = (lambda_param ** (i - 1))

    # Define long-term correlation calculation
    
    #MV = MV.values
    if MV is not None and RC is not None:         
        for i in range(1, K + 1):
            m_rc += theta_rc * phi[i] * RC[i]
            m1 += theta1 * phi[i] * MV[i] 
            lt_rho = m_rc + m1
    
    if MV is not None and RC is None:
        for i in range(1, K + 1):
            m1 +=theta1 * phi[i] * MV[i]  
            lt_rho = m1
    
    if MV is None and RC is not None:
        for i in range(1, K + 1):
            m_rc += theta_rc * phi[i] * RC[i]
            lt_rho = m_rc 
    
    lt_rho = (np.exp(2 * lt_rho) - 1) / (np.exp(2 * lt_rho) + 1)  # Fisher transformation

    q_11[0] = 1
    q_12[0] = (np.mean(z1 * z2) - np.mean(z1) * np.mean(z2)) / \
               (np.std(z1) * np.std(z2))
    q_22[0] = 1

    garchln = np.zeros(T)

    # Main loop for calculating log-likelihood based on GARCH model
    for i in range(1, T):
        q_11[i] = (1 - alpha - beta) * q_11[0] + alpha * z1[i - 1] ** 2 + beta * q_11[i - 1]
        q_22[i] = (1 - alpha - beta) * q_22[0] + alpha * z2[i - 1] ** 2 + beta * q_22[i - 1]
        q_12[i] = (1 - alpha - beta) * lt_rho[i] + alpha * z1[i - 1] * z2[i - 1] + beta * q_12[i - 1]
        
        rho[i] = q_12[i] / np.sqrt(q_11[i] * q_22[i])
        
        # Calculate the log-likelihood
        garchln[i] = -np.log(1 - rho[i] ** 2) - (1 / (1 - rho[i] ** 2)) * (
            z1[i] ** 2 - 2 * rho[i] * z1[i] * z2[i] + z2[i] ** 2
        )
        
        # garchln[i] = -np.log(1 - rho[i] ** 2) - (1 / (1 - rho[i] ** 2)) * (
        #     z1[i] ** 2 - 2 * rho[i] * z1[i] * z2[i] + z2[i] ** 2
        # )

    return -np.sum(garchln[1:])  # Return the sum of log-likelihood values, excluding the first observation

def dcc_rc_x(params, udata_list, K, lambda_param, MV = None, RC = None):
    """
    Calculate the log-likelihood for a GARCH(1,1) model with an additional term for macroeconomic variables.

    Similar to the previous log-likelihood function but includes additional terms for lagged correlations.
    """
    # Similar logic as loglikelihood_garch, with additional steps for macroeconomic variables (MV)
   # Verificação inicial para garantir que pelo menos MV ou RC seja fornecido
    if MV is None and RC is None:
        print("Error: Both MV and RC are missing. Please provide at least one of them to proceed.")
        raise ValueError("Both MV and RC are missing. Provide at least one of these inputs.")

    if MV is not None and RC is not None:
        alpha, beta, theta1, theta_rc = params

    if MV is not None and RC is None:
        alpha, beta, theta1 = params
    
    if MV is None and RC is not None:
        alpha, beta, theta_rc = params
    
    #Apagar depois
    # alpha = 0.05
    # beta = 0.9
    # theta1 = 0.1
    # theta_rc = 0.1
    # b = 'Yes'
    # X = "Yes"

    T = len(udata_list[0])
    q_11 = np.zeros(T)
    q_12 = np.zeros(T)
    q_22 = np.zeros(T)
    lt_rho = np.zeros(T)
    m1 = np.zeros(T)
    m_rc = np.zeros(T)
    rho = np.zeros(T)
    
    z1 = udata_list[0]  # Residuals of S&P 500
    z2 = udata_list[1]  # Residuals of US Bond

    # Initialize weights for lagged realized correlations
    phi = np.zeros(K + 1)
    for i in range(1, K + 1):
        phi[i] = (lambda_param ** (i - 1))

    # Define long-term correlation calculation
    
    #MV = MV.values
    if MV is not None and RC is not None:         
        for i in range(1, K + 1):
            m_rc += theta_rc * phi[i] * RC[i]
            m1 += theta1 * phi[i] * MV[i] 
            lt_rho = m_rc + m1
    
    if MV is not None and RC is None:
        for i in range(1, K + 1):
            m1 +=theta1 * phi[i] * MV[i]  
            lt_rho = m1
    
    if MV is None and RC is not None:
        for i in range(1, K + 1):
            m_rc += theta_rc * phi[i] * RC[i]
            lt_rho = m_rc 
    
    lt_rho = (np.exp(2 * lt_rho) - 1) / (np.exp(2 * lt_rho) + 1)  # Fisher transformation

    q_11[0] = 1
    q_12[0] = (np.mean(z1 * z2) - np.mean(z1) * np.mean(z2)) / \
               (np.std(z1) * np.std(z2))
    q_22[0] = 1

    garchln = np.zeros(T)

    # Main loop for calculating log-likelihood based on GARCH model
    for i in range(1, T):
        q_11[i] = (1 - alpha - beta) * q_11[0] + alpha * z1[i - 1] ** 2 + beta * q_11[i - 1]
        q_22[i] = (1 - alpha - beta) * q_22[0] + alpha * z2[i - 1] ** 2 + beta * q_22[i - 1]
        q_12[i] = (1 - alpha - beta) * lt_rho[i] + alpha * z1[i - 1] * z2[i - 1] + beta * q_12[i - 1]
        
        rho[i] = q_12[i] / np.sqrt(q_11[i] * q_22[i])
        
        # Calculate the log-likelihood
        garchln[i] = -np.log(1 - rho[i] ** 2) - (1 / (1 - rho[i] ** 2)) * (
            z1[i] ** 2 - 2 * rho[i] * z1[i] * z2[i] + z2[i] ** 2
        )
        
        # garchln[i] = -np.log(1 - rho[i] ** 2) - (1 / (1 - rho[i] ** 2)) * (
        #     z1[i] ** 2 - 2 * rho[i] * z1[i] * z2[i] + z2[i] ** 2
        # )

    return np.sum(garchln[1:]), q_11, q_22, q_12, rho, garchln, lt_rho, m1, m_rc

def correlation_ratio (lt_rho, rho, m1 = None, m_rc = None):
    
    cr1 =100 * np.var(lt_rho) / np.var(rho) 
    cr2 = 100 * np.var(m1)/np.var(m_rc)

    return cr1, cr2


def calculate_criteria(log_likelihood, num_params, T):
    """
    Calculate AIC, BIC, and HQC given the log-likelihood, number of parameters, and number of observations.
    """
    AIC = -2 * log_likelihood + 2 * num_params
    BIC = -2 * log_likelihood + np.log(T) * num_params
    HQC = -2 * log_likelihood + 2 * np.log(np.log(T)) * num_params
    
    return {'AIC': AIC, 'BIC': BIC, 'HQC': HQC}

def QMLE(params, udata_list, K, MV, lambda_param, log_likelihood):
    """
    Estimate the parameters using QMLE (Quasi Maximum Likelihood Estimation), returning the log-likelihood, errors, and p-values.
    
    Args:
    params (np.ndarray): The parameters of the model [alpha, beta].
    udata_list (list): The list of standardized residuals.
    K (int): Number of lags for macro variables.
    MV (np.ndarray): Macro variables data.
    lambda_param (float): Exponential weight decay parameter.
    hessian (np.ndarray): The Hessian matrix (second derivative of log-likelihood).
    log_likelihood (float): The log-likelihood value calculated from the model.
    
    Returns:
    dict: A dictionary containing the log-likelihood, parameter standard errors, and p-values.
    """

    params=opt_params 


    # Number of parameters in the model
    num_params = len(params)
    
    # Number of observations (T)
    T = len(udata_list[0])
    
    # Calculate information criteria (AIC, BIC, HQC)
    criteria = calculate_criteria(log_likelihood, num_params, T)
        
    # Calculate standard errors (square root of diagonal elements of the inverse Hessian)
    def func_to_optimize(params):
        return loglikelihood_garch(params, udata_list, K, MV, lambda_param)
    # Criação do objeto Hessian usando numdifftools
    hess = ndt.Hessian(func_to_optimize)
    # Calcular a Hessiana numericamente no ponto ótimo encontrado
    hessian = hess(params)

    hessian_inv = -np.linalg.inv(hessian)
    mhess = -hessian_inv
    std_errors = np.sqrt(np.diagonal(mhess))
    
    # Calculate p-values using the normal distribution and standard errors
    p_values = 2 * (1 - norm.cdf(np.abs(params / std_errors)))  # Calculate p-values based on Z-scores
    
    return {
        'log_likelihood': log_likelihood,
        'AIC': criteria['AIC'],
        'BIC': criteria['BIC'],
        'HQC': criteria['HQC'],
        'params': params,
        'std_errors': std_errors,
        'p_values': p_values,  # Include gradient for transparency
        'hessian': hessian     # Include Hessian for transparency
    }

def corr_matrix(rho):
    """
    Generates a list of 2x2 correlation matrices for each time step, based on the provided correlation values.

    The function creates a correlation matrix for each time step, where:
    - Rt_11[i] = 1, representing the diagonal element (variance) of the first variable in the covariance matrix at time i.
    - Rt_12[i] = rho[i], representing the off-diagonal element (covariance) between the first and second variable at time i.
    - Rt_21[i] = rho[i], representing the off-diagonal element (covariance) between the first and second variable at time i (symmetric with Rt_12).
    - Rt_22[i] = 1, representing the diagonal element (variance) of the second variable at time i.

    The function returns a list `Rt` containing these 2x2 correlation matrices for each time step in `rho`.

    Args:
    rho (np.ndarray): A 1D array containing the correlation values between two variables at each time step.

    Returns:
    Rt (list): A list of 2x2 correlation matrices for each time step, constructed from the values of `rho`.
    """

    # Initialize the vectors for the elements of the covariance matrices
    Rt_11 = np.zeros(len(rho))
    Rt_12 = np.zeros(len(rho))
    Rt_21 = np.zeros(len(rho))
    Rt_22 = np.zeros(len(rho))
    
    # Initialize a list to store the covariance matrices for each time step
    Rt = []

    # Loop to fill in the values of each covariance matrix at each time step
    for i in range(len(rho)): 
        # Assign values to individual variables for each time step
        Rt_11[i] = 1
        Rt_12[i] = rho[i]
        Rt_21[i] = rho[i]
        Rt_22[i] = 1

        # Create the 2x2 covariance matrix for time step i
        Rt_matrix = np.array([[Rt_11[i], Rt_12[i]],
                            [Rt_21[i], Rt_22[i]]])
    
        # Store the matrix in the list
        Rt.append(Rt_matrix)

    return Rt 

def std_dev_matrix(udata_list):
    """
    Generates a list of 2x2 standard deviation matrices for each time step based on the residuals.

    Args:
    udata_list (list): A list containing two arrays, z1 and z2, which are the residuals of S&P 500 and US Bond returns.

    Returns:
    Dt (list): A list of 2x2 standard deviation matrices for each time step.
    """

    # Extract the residuals from the input list
    z1 = udata_list[0]  # Residuals of S&P 500
    z2 = udata_list[1]  # Residuals of US Bond

    # Initialize arrays to store the individual elements of the standard deviation matrices
    Dt_11 = np.zeros(len(z1))  # Diagonal element for the first variable (z1)
    Dt_12 = np.zeros(len(z1))  # Off-diagonal element (zero for standard deviation matrix)
    Dt_21 = np.zeros(len(z1))  # Off-diagonal element (zero for standard deviation matrix)
    Dt_22 = np.zeros(len(z1))  # Diagonal element for the second variable (z2)

    # Initialize the list to store the 2x2 matrices
    Dt = []

    # Loop through each time step to calculate the standard deviation matrix
    for i in range(len(z1)): 
        # Compute the diagonal elements as the square root of residuals
        Dt_11[i] = np.sqrt(z1[i])  # Standard deviation for z1
        Dt_12[i] = 0               # Off-diagonal element is zero
        Dt_21[i] = 0               # Off-diagonal element is zero
        Dt_22[i] = np.sqrt(z2[i])  # Standard deviation for z2

        # Construct the 2x2 standard deviation matrix for the current time step
        Rt_matrix = np.array([[Dt_11[i], Dt_12[i]],
                              [Dt_21[i], Dt_22[i]]])
    
        # Append the matrix to the list
        Dt.append(Rt_matrix)

    # Return the list of 2x2 standard deviation matrices
    return Dt

def cov_matrix(Rt, Dt):
    """
    Calculates covariance matrices by multiplying `Dt`, `Rt`, and `Dt` again for each time step.

    Args:
        Rt (list): List of 2x2 correlation matrices.
        Dt (list): List of 2x2 standard deviation matrices.

    Returns:
        list: List of 2x2 covariance matrices.
    """
    cov_matrices = []
    
    for i in range(len(Rt)):
        cov_matrix = Dt[i] @ Rt[i] @ Dt[i]  
        cov_matrices.append(cov_matrix)

    return cov_matrices

def efficient_frontier(expected_returns, cov_matrix, target_returns):
    """
    Constructs the efficient frontier for a portfolio based on expected returns and a covariance matrix.

    Args:
        expected_returns (ndarray): Expected returns for each asset.
        cov_matrix (ndarray): Covariance matrix of asset returns.
        target_returns (ndarray): Array of target returns to achieve.

    Returns:
        tuple: A tuple containing:
            - efficient_frontier_weights (list): List of optimal weights for each portfolio on the efficient frontier.
            - portfolios (DataFrame): DataFrame storing weights for all assets at each portfolio.
            - efficient_frontier_risk (list): List of portfolio risks (standard deviations) for each target return.
            - efficient_frontier_returns (list): List of achieved returns for each portfolio on the efficient frontier.
    """
    # Helper function to calculate portfolio variance (risk)
    def portfolio_variance(weights):
        return weights.dot(cov_matrix).dot(weights.T)
    
    # Helper function to calculate portfolio return
    def portfolio_return(weights):
        return weights.dot(expected_returns)

    # Lists to store the results of the efficient frontier
    efficient_frontier_weights = []  # Stores weights for each portfolio
    efficient_frontier_risk = []  # Stores risk for each portfolio
    efficient_frontier_returns = []  # Stores returns for each portfolio
    portfolios = pd.DataFrame(index=tickers)  # DataFrame to store portfolio weights

    # Loop through each target return to construct the efficient frontier
    for target_return in target_returns:
        # Constraint: Ensure the portfolio meets the target return
        # Constraint: Ensure the portfolio weights sum to 1
        constraints = (
            {'type': 'eq', 'fun': lambda weights: weights.dot(expected_returns) - target_return},
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        )
               
        # Initialize weights evenly among assets
        assets_number = len(tickers)  # Number of assets
        initial_weights = [1 / assets_number for i in range(assets_number)]  # Equal weights initially
        
        # Bounds to ensure weights are between 0 and 1
        bnds = [[0, 1] for _ in range(assets_number)]
        
        # Optimization to minimize portfolio variance for the given target return
        result = minimize(portfolio_variance, initial_weights, method='SLSQP', constraints=constraints, bounds=bnds)
        
        # Extract the optimal weights for the portfolio
        w_frontier = result.x
        efficient_frontier_weights.append(w_frontier)
        
        # Calculate and record the portfolio risk (standard deviation)
        port_risk = portfolio_variance(w_frontier)
        port_risk = np.sqrt(port_risk)  # Convert variance to standard deviation
        efficient_frontier_risk.append(port_risk)
        
        # Calculate and record the portfolio return
        port_return = portfolio_return(w_frontier)
        efficient_frontier_returns.append(port_return)
    
        # Store the portfolio weights in the DataFrame
        portfolios[round(port_risk, 2)] = w_frontier
        portfolios = round(portfolios, 2)  # Round weights for clarity

    # Return all results as a tuple
    return efficient_frontier_weights, portfolios, efficient_frontier_risk, efficient_frontier_returns


def portfolio_performance(selected_portfolio):
    """
    Calculates the cumulative return and Sharpe ratio for a selected portfolio.

    Args:
        selected_portfolio (float): The risk (standard deviation) of the selected portfolio 
                                    used as the denominator in Sharpe ratio calculation.

    Returns:
        tuple: 
            - portfolio_return_acc (Series): Cumulative return of the selected portfolio over time.
            - sharpe_ratio (float): The Sharpe ratio of the selected portfolio.
    """

    # Calculate the weighted return for each asset in the portfolio
    # The weights are taken from the selected portfolio in the `portfolios` DataFrame
    weighted_return = portfolios[selected_portfolio] * rets

    # Sum the weighted returns across all assets to get the portfolio return for each time step
    portfolio_return = weighted_return.sum(axis=1)

    # Calculate the cumulative product of (1 + daily portfolio returns) to get cumulative returns over time
    portfolio_return_acc = (1 + portfolio_return).cumprod()

    # Portfolio risk (provided as `selected_portfolio`)
    portfolio_risk = selected_portfolio

    # Calculate the annualized mean return of the portfolio (assuming 252 trading days per year)
    annual_return = portfolio_return.mean() * 252

    # Calculate the Sharpe ratio (risk-adjusted return)
    # In this implementation, the risk-free rate (rf) is omitted; it can be reintroduced if needed
    sharpe_ratio = (annual_return) / portfolio_risk

    return portfolio_return_acc, sharpe_ratio

#%% Analysis

# Starting point of the analysis
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2020, 12, 30)

# Define tickers for S&P 500 and US Treasury Bond yield
tickers = ['^GSPC', '^TNX']

# Download adjusted close prices for the specified tickers
close_prices = yf.download(tickers, start=start, end=end)['Adj Close']

# Calculate daily returns from the adjusted close prices
rets = ((close_prices / close_prices.shift(1)) - 1).dropna() * 100  # Returns in percentage

# # Download CPI data from FRED (Federal Reserve Economic Data)
# cpi_data = web.DataReader('CPIAUCNS', 'fred', start, end)
# cpi_data = np.log(cpi_data).diff()  # Calculate log differences to obtain inflation rate
# cpi_data = cpi_data.dropna()  # Drop missing values

cpi_data= pd.read_excel('CPIAUCNS.xlsx')

# Supondo que o DataFrame seja chamado df
cpi_data['observation_date'] = pd.to_datetime(cpi_data['observation_date']) 
cpi_data.rename(columns={'observation_date': 'Date'}, inplace=True) # Converte para datetime
cpi_data.set_index('Date', inplace=True) # Calculate log differences to obtain inflation rate
cpi_data = np.log(cpi_data).diff()  # Calculate log differences to obtain inflation rate
cpi_data = cpi_data.dropna()  # Drop missing values

# Define the macroeconomic variables (CPI data)
MV = cpi_data
MV = MV.resample('D').ffill()  #Resample MV data to daily frequency using forward-fill

# Merge the two dataframes on the common dates
MV.index = pd.to_datetime(MV.index).tz_localize(None)
rets.index = pd.to_datetime(rets.index).tz_localize(None)

merged_data = pd.merge(MV, rets, left_index=True, right_index=True, how='inner')
MV = merged_data['CPIAUCNS']
rets['^GSPC'] = merged_data['^GSPC']
rets['^TNX'] = merged_data['^TNX']

# Drop columns with missing values (NaN)
#merged_data_cleaned = merged_data.dropna(axis=1, how='any')

# Initialize dictionaries for storing model parameters and uniform data
model_parameters = {}
udata_list = []

# Run GARCH model on the returns data
udata_list, model_parameters = run_garch_on_return(rets, udata_list, model_parameters)

z1 = udata_list[0]  # Residuals of S&P 500
z2 = udata_list[1]  # Residual of US Bonds

residuals_df = pd.DataFrame({"z1": z1, "z2": z2}, index=rets.index)

# Resample para mensal e calcule a correlação entre z1 e z2 para cada mês
RC_df = residuals_df.resample('MS').apply(lambda x: x['z1'].corr(x['z2']))
RC = RC_df

# Plot the conditional volatility of the fitted GARCH models
plot_conditional_volatility(rets, model_parameters)

# Optimize the log-likelihood to find the best-fitting parameters
initial_params_dcc_x = np.array([0.05, 0.9, 0.0])  # Initial guess for parameters [alpha, beta, theta1]
initial_params_dcc_rc_x = np.array([0.05, 0.9, 0.0, 0.0])  # Initial guess for parameters [alpha, beta, theta1, theta_rc]
initial_params_dcc_rc = np.array([0.05, 0.9, 0.0]) # Initial guess for parameters [alpha, beta, theta_rc]

lambda_param = 0.96  # Decay parameter for exponential smoothing
K = 48  # Number of lags for the macroeconomic variables

# Define constraints and bounds for the optimization
cons = ({'type': 'ineq', 'fun': lambda x: -x[0] - x[1] + 1})  # Constraints on alpha and beta
bnds_dcc_x = ((0.0001, 0.5), (0.0001, 0.9997), (0, 0.9997))  # Bounds for alpha, beta and theta1 parameters
bnds_dcc_rc_x = ((0.0001, 0.5), (0.0001, 0.9997), (0, 0.9997), (0, 0.9997)) # Bounds for alpha, beta, theta1 and theta_rc parameters
bnds_dcc_rc = ((0.0001, 0.5), (0.0001, 0.9997), (0, 0.9997))  # Bounds for alpha, beta and theta_rc parameters

# Minimize the log-likelihood function to estimate the optimal parameters
# opt_out = minimize(loglikelihood_garch, initial_params, args=(udata_list, K, MV, lambda_param), bounds=bnds, constraints=cons, method = 'L-BFGS-B')

# opt_out = minimize(loglikelihood_garch, initial_params, args=(udata_list, K, MV, lambda_param), bounds=bnds, method = 'BFGS', tol=1e-3, options={'maxiter': 500, 'disp': True})
# print(opt_out)
opt_out = minimize(loglikelihood_garch, initial_params_dcc_rc_x, args=(udata_list, K, lambda_param, MV, RC), bounds=bnds_dcc_rc_x , constraints=cons, method = 'SLSQP')

# Extract optimized parameters and log-likelihood
opt_params = opt_out.x
opt_loglikehood = opt_out.fun
#opt_hessian = opt_out.hess_inv
#opt_hessian = opt_out.hess_inv.todense()

# Get the log-likelihood, q_11, q_22, q_12, and rho from the dcc_rc_x function
log_likelihood, q_11, q_22, q_12, rho, garch_ln_matrix, lt_rho, m1, m_rc  = dcc_rc_x(opt_params, udata_list, K, lambda_param, MV, RC)

# Calculate the information criteria (AIC, BIC, HQC) based on the optimized log-likelihood
information_criteria = calculate_criteria(log_likelihood=opt_loglikehood, num_params=len(opt_params), T=len(udata_list[0]))

# Estimate the model parameters using QMLE (Quasi Maximum Likelihood Estimation)
#results = QMLE(params=opt_params, udata_list=udata_list, K=K, MV=MV, lambda_param=lambda_param, hessian=opt_hessian, log_likelihood=log_likelihood)

# Print the results: log-likelihood, AIC, BIC, HQC, parameters, standard errors, and p-values
# print("Log-Likelihood:", results['log_likelihood'])
# print("AIC:", results['AIC'])
# print("BIC:", results['BIC'])
# print("HQC:", results['HQC'])
# print("Parameters:", results['params'])
# print("Standard Errors:", results['std_errors'])
# print("P-values:", results['p_values'])

#%% Portfolio Optimization 
# Calculate daily covariance matrices Ht
Rt = corr_matrix(rho)  # Calculate correlation matrices for each time step
Dt = std_dev_matrix(udata_list)  # Calculate standard deviation matrices for each time step
Ht = cov_matrix(Rt, Dt)  # Calculate covariance matrices using Rt and Dt

# Portfolio optimization
expected_returns = 252 * rets.mean()  # Calculate annualized expected returns

# Define target returns for portfolio optimization
# Target returns represent desired portfolio returns for optimization
target_returns = np.arange(0.0, 2.0, 0.04)

# Compute the efficient frontier (optimal portfolios)
efficient_frontier_weights, portfolios, efficient_frontier_risk, efficient_frontier_returns = efficient_frontier(
    expected_returns, Ht[1], target_returns)

# Plot the efficient frontier
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(efficient_frontier_risk, efficient_frontier_returns, '-')  # Plot risk vs. return
plt.title('Efficient Frontier')  # Title of the plot
plt.xlabel('Risk (Portfolio Standard Deviation)')  # X-axis label
plt.ylabel('Expected Return')  # Y-axis label
plt.grid(True)  # Add grid lines
plt.show()  # Display the plot

# Select a specific portfolio by its risk value
selected_portfolio = 0.81  # Specify the desired risk level for the portfolio

# Calculate the accumulated portfolio return and Sharpe ratio
portfolio_return_acc, sharpe_ratio = portfolio_performance(selected_portfolio)

# Plot the accumulated portfolio returns over time
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(portfolio_return_acc)  # Plot the accumulated portfolio returns
plt.title('Portfolio Accumulated Return')  # Title of the plot
plt.xlabel('Time')  # X-axis label
plt.ylabel('Accumulated Return')  # Y-axis label
plt.show()  # Display the plot