import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import  het_breuschpagan
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def process_wpi_variation_data(file_path):
    try:
        # Read the entire Excel file
        raw_data = pd.read_excel(file_path, header=None)
    except Exception as e:
        print(f"Error loading WPI data: {e}")
        raise
    
    # Initialize empty dataframe to store combined results
    combined_data = pd.DataFrame(columns=['Year', 'WPI'])
    
    # Process each base year section
    current_section = None
    header_row_idx = None
    year_col_idx = None
    ac_col_idx = None
    data_rows = []
    
    # Iterate through the rows to identify sections and extract data
    for idx, row in raw_data.iterrows():
        row_str = ' '.join([str(x) for x in row if pd.notna(x)])
        
        # Check if this is a base year header row
        if 'Base' in row_str and ('2011-12' in row_str or '2004-05' in row_str or '1993-94' in row_str or '1981-82' in row_str or '1970-71' in row_str):
            # Process previous section if exists
            if current_section and header_row_idx and year_col_idx is not None and ac_col_idx is not None and data_rows:
                section_df = pd.DataFrame(data_rows, columns=['Year', 'WPI'])
                combined_data = pd.concat([combined_data, section_df])
                data_rows = []
            
            # Start new section
            current_section = row_str
            header_row_idx = None
            year_col_idx = None
            ac_col_idx = None
            continue
        
        # Check if this is a column header row (contains "Year" and "AC")
        if 'Year' in row_str and ('AC' in row_str ):
            header_row_idx = idx
            
            # Find Year and AC column indices
            for i, val in enumerate(row):
                if pd.notna(val) and 'Year' in str(val):
                    year_col_idx = i
                elif pd.notna(val) and ('AC' in str(val) or val == 2):
                    ac_col_idx = i
            continue
        
        # If we have identified headers, extract data rows
        if header_row_idx and year_col_idx is not None and ac_col_idx is not None:
            # Skip the row with column numbers (1, 2, 3, etc.)
            if all(isinstance(x, (int, float)) and x in range(1, 10) for x in [val for val in row if pd.notna(val)]):
                continue
                
            # Check if this is a data row (has year in YYYY-YY format)
            year_val = row[year_col_idx]
            ac_val = row[ac_col_idx]
            
            if pd.notna(year_val) and pd.notna(ac_val) and isinstance(year_val, str) and '-' in year_val:
                # Extract first year from format like "2023-24"
                year = int(year_val.split('-')[0])
                data_rows.append([year, ac_val])
    
    # Process the last section
    if current_section and header_row_idx and year_col_idx is not None and ac_col_idx is not None and data_rows:
        section_df = pd.DataFrame(data_rows, columns=['Year', 'WPI'])
        combined_data = pd.concat([combined_data, section_df])
    
    # Remove duplicates (in case of overlapping years from different base periods)
    combined_data = combined_data.drop_duplicates(subset='Year', keep='first')
    
    # Sort by year
    combined_data = combined_data.sort_values('Year')
    
    # Set Year as index
    combined_data.set_index('Year', inplace=True)
    combined_data = combined_data.loc[1991:2017]
    # Check if we have data
    if combined_data.empty:
        raise ValueError("No WPI data could be extracted from the file")
    return combined_data

def parse_rbi_policy_rates(file_path):
    try:
        data = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading RBI policy rates data: {e}")
        raise
    
    # Check if the data has headers
    if 'Effective Date' in data.columns:
        # Data already has proper headers
        policy_data = data
    else:
        # Consider adding header detection logic here
        raise ValueError("Data does not have expected headers")
    
    # Clean column names
    policy_data.columns = [str(col).strip() for col in policy_data.columns]
    
    # Identify the required columns
    date_col = [col for col in policy_data.columns if 'Effective Date' in col][0]
    bank_rate_col = [col for col in policy_data.columns if 'Bank Rate' in col][0]
    crr_col = [col for col in policy_data.columns if 'Cash Reserve' in col or 'CRR' in col][0]
    slr_col = [col for col in policy_data.columns if 'Statutory Liquidity' in col or 'SLR' in col][0]
    
    # Convert date to datetime
    policy_data[date_col] = pd.to_datetime(policy_data[date_col], format='%d-%m-%Y', errors='coerce')
    
    # Extract year from date
    policy_data['Year'] = policy_data[date_col].dt.year
    
    # Convert rate columns to numeric, handling dashes or empty cells
    for col in [bank_rate_col, crr_col, slr_col]:
        policy_data[col] = pd.to_numeric(policy_data[col], errors='coerce')
    
    # For each year, get the latest policy rates
    # Group by year and get the most recent values for each rate
    yearly_rates = policy_data.sort_values(date_col).groupby('Year').last()
    
    # Create a clean dataframe with required columns
    result = pd.DataFrame({
        'BANK_RATE': yearly_rates[bank_rate_col],
        'CRR': yearly_rates[crr_col],
        'SLR': yearly_rates[slr_col]
    })
    
    # Handle missing years by reindexing and filling missing years
    # Create full range of years from min to max in the data
    min_year = min(result.index.min(), 1991)  # Ensure we go back to at least 1991
    max_year = max(result.index.max(), 2017)  # Ensure we go up to at least 2017
    full_years = pd.Index(range(min_year, max_year + 1))
    result = result.reindex(full_years)
    
    # For missing years, fill by forward fill (carry last known value)
    # This is the most economically sound approach for policy rates
    result = result.fillna(method='ffill')
    
    # For any remaining missing values at the start, fill by backward fill
    # This handles cases where the earliest available data is after 1991
    result = result.fillna(method='bfill')
    
    # Check if we have data
    if result.empty:
        raise ValueError("No policy rate data could be extracted")
    
    # Filter to years 1991 to 2017 for the paper analysis
    final_result = result.loc[1991:2017]
    
    return final_result

def run_regression_model(data):
    X = data[['BANK_RATE', 'CRR', 'SLR']]
    X = sm.add_constant(X)
    y = data['WPI']
    
    model = sm.OLS(y, X).fit()
    return model

def breusch_godfrey_lm_test(model, lags=13):
    """Perform Breusch-Godfrey serial correlation LM test."""
    bg_test = acorr_breusch_godfrey(model, nlags=lags)
    
    return {
        'lm_stat': bg_test[0],
        'p_value': bg_test[1],
        'f_stat': bg_test[2],
        'f_p_value': bg_test[3]
    }

def breusch_pagan_godfrey_test(model):
    """Perform Breusch-Pagan-Godfrey test for heteroscedasticity."""
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    
    return {
        'lm_stat': bp_test[0],
        'lm_p_value': bp_test[1],
        'f_stat': bp_test[2],
        'f_p_value': bp_test[3]
    }

def jarque_bera_test(residuals):
    """Perform Jarque-Bera test for normality of residuals."""
    jb_test = jarque_bera(residuals)
    
    return {
        'jb_stat': jb_test[0],
        'jb_p_value': jb_test[1],
        'skewness': jb_test[2],
        'kurtosis': jb_test[3]
    }

def perform_diagnostic_tests(model):
    residuals = model.resid
    
    bg_results = breusch_godfrey_lm_test(model)
    bp_results = breusch_pagan_godfrey_test(model)
    jb_results = jarque_bera_test(residuals)
    
    return {
        'autocorrelation': bg_results,
        'heteroscedasticity': bp_results,
        'normality': jb_results,
        'residuals': residuals
    }

def check_stationarity(data):
    results = {}
    
    for column in data.columns:
        # ADF test at level with constant
        adf_level = adfuller(data[column], maxlag=9, regression='c')
        
        # ADF test at first difference with constant
        first_diff = data[column].diff().dropna()
        adf_diff = adfuller(first_diff, maxlag=9, regression='c')
        
        # Calculate Durbin-Watson statistic
        dw_level = sm.stats.stattools.durbin_watson(
            sm.OLS(data[column], sm.add_constant(np.arange(len(data[column])))).fit().resid
        )
        
        dw_diff = sm.stats.stattools.durbin_watson(
            sm.OLS(first_diff, sm.add_constant(np.arange(len(first_diff)))).fit().resid
        )
        
        results[column] = {
            'level': {
                'p_value': adf_level[1],
                'statistic': adf_level[0],
                'critical_values': adf_level[4],
                'durbin_watson': dw_level
            },
            'first_difference': {
                'p_value': adf_diff[1],
                'statistic': adf_diff[0],
                'critical_values': adf_diff[4],
                'durbin_watson': dw_diff
            }
        }
    
    return results

def johansen_cointegration_test(data):
    # Johansen test
    johansen_results = coint_johansen(data.values, det_order=0, k_ar_diff=1)
    
    # Extract trace statistics and eigenvalues
    trace_stats = johansen_results.lr1
    eigen_stats = johansen_results.lr2
    
    # Extract critical values
    crit_vals_trace = johansen_results.cvt
    crit_vals_eigen = johansen_results.cvm
    
    # Create summary DataFrame for trace test
    trace_summary = pd.DataFrame({
        'Hypothesized No. of CE(s)': ['None', 'At most 1', 'At most 2', 'At most 3'][:data.shape[1]],
        'Eigenvalue': johansen_results.eig,
        'Trace Statistic': trace_stats,
        '0.05 Critical Value': crit_vals_trace[:, 1],
        'p-value': [stats.chi2.sf(trace_stats[i], data.shape[1] - i) for i in range(data.shape[1])]
    })
    
    # Create summary DataFrame for eigenvalue test
    eigen_summary = pd.DataFrame({
        'Hypothesized No. of CE(s)': ['None', 'At most 1', 'At most 2', 'At most 3'][:data.shape[1]],
        'Eigenvalue': johansen_results.eig,
        'Max-Eigen Statistic': eigen_stats,
        '0.05 Critical Value': crit_vals_eigen[:, 1],
        'p-value': [stats.chi2.sf(eigen_stats[i], 1) for i in range(data.shape[1])]
    })
    
    return {
        'trace_summary': trace_summary,
        'eigen_summary': eigen_summary,
        'raw_results': johansen_results
    }

def build_ecm_model(data):
    """
    Build Error Correction Model (ECM) to analyze short and long-term relationships.
    
    ECM equation: DWPI = β1 + β2D*bank-rate + β3D*CRR + β4D*SLR + β5ut-1 + v
    """
    # Step 1: Estimate long-run relationship (cointegrating equation)
    X = sm.add_constant(data[['BANK_RATE', 'CRR', 'SLR']])
    y = data['WPI']
    
    long_run_model = sm.OLS(y, X).fit()
    
    # Step 2: Get residuals (error correction term)
    residuals = long_run_model.resid
    
    # Step 3: Create first differences for all variables
    data_diff = data.diff().dropna()
    
    # Rename columns to indicate differenced variables
    data_diff.columns = ['D' + col for col in data_diff.columns]
    
    # Add lagged residuals to the differenced data (error correction term)
    data_diff['U1_lag1'] = residuals.shift(1).loc[data_diff.index]
    
    # Drop NaN values
    data_diff = data_diff.dropna()
    
    # Step 4: Estimate the ECM model
    X_ecm = sm.add_constant(data_diff[['DBANK_RATE', 'DCRR', 'DSLR', 'U1_lag1']])
    y_ecm = data_diff['DWPI']
    
    ecm_model = sm.OLS(y_ecm, X_ecm).fit()
    
    # Calculate Durbin-Watson statistic for ECM model
    dw_stat = sm.stats.stattools.durbin_watson(ecm_model.resid)
    
    # Check if model is spurious (R² > DW)
    r_squared = ecm_model.rsquared
    is_spurious = r_squared > dw_stat
    
    return {
        'model': ecm_model,
        'residuals': ecm_model.resid,
        'r_squared': r_squared,
        'durbin_watson': dw_stat,
        'is_spurious': is_spurious
    }

def plot_time_series(data, title="Time Series Plot"):
    plt.figure(figsize=(14, 8))
    
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.xticks(list(data.index)[::2])  # Show every other year
    plt.savefig(title)

def plot_residual_histogram(residuals, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, stat='density')
    # Plot normal curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, 'k', linewidth=2)
    # Add Jarque-Bera test results
    jb_results = jarque_bera_test(residuals)
    jb_stat = jb_results['jb_stat']
    jb_p_value = jb_results['jb_p_value']
    plt.title(f'{title}\nJB Stat: {jb_stat:.4f}, p-value: {jb_p_value:.4f}')
    plt.grid(True)
    plt.savefig(title)

def main():    
    wpi_data = process_wpi_variation_data("HBS Table No. 230 _ Wholesale Price Index - Annual Variation.xlsx")
    rbi_data = parse_rbi_policy_rates("HBS Table No. 43 _ Major Monetary Policy Rates and Reserve Requirements - Bank Rate, LAF (Repo, Reverse Repo and MSF) Rates, CRR & SLR.xlsx")
    data = pd.concat([wpi_data, rbi_data], axis=1)
    
    plot_time_series(data, "WPI Annual Variation (%) and RBI Policy Rates (1991-2017)")
    
    model = run_regression_model(data)
    print("\nRegression Model Results:")
    print(model.summary())
    
    diagnostic_tests = perform_diagnostic_tests(model)
    
    print("\nDiagnostic Tests Results:")
    print(f"Breusch-Godfrey LM Test p-value: {diagnostic_tests['autocorrelation']['p_value']:.4f}")
    print(f"Breusch-Pagan-Godfrey Test p-value: {diagnostic_tests['heteroscedasticity']['lm_p_value']:.4f}")
    print(f"Jarque-Bera Test p-value: {diagnostic_tests['normality']['jb_p_value']:.4f}")
    
    # Plot residuals of original model
    plt.figure(figsize=(12, 6))
    plt.plot(diagnostic_tests['residuals'])
    plt.title('Residuals of the Regression Model')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.grid(True)
    plt.savefig('Residuals of the Regression Model')
    
    plot_residual_histogram(diagnostic_tests['residuals'], 'Histogram of Residuals')
    
    # Check stationarity
    stationarity_results = check_stationarity(data)
    print("\nStationarity Test Results (ADF Test):")
    for var, results in stationarity_results.items():
        print(f"\nVariable: {var}")
        print(f"At Level - p-value: {results['level']['p_value']:.4f}")
        print(f"At First Difference - p-value: {results['first_difference']['p_value']:.4f}")
    
    # Johansen cointegration test
    johansen_results = johansen_cointegration_test(data)
    print("\nJohansen Cointegration Test Results:")
    print("\nTrace Test:")
    print(johansen_results['trace_summary'])
    print("\nEigenvalue Test:")
    print(johansen_results['eigen_summary'])
    
    # Build ECM model
    ecm_results = build_ecm_model(data)
    print("\nError Correction Model Results:")
    print(ecm_results['model'].summary())
    
    # Check if ECM model is spurious
    print(f"\nECM Model - R²: {ecm_results['r_squared']:.4f}, DW: {ecm_results['durbin_watson']:.4f}")
    print(f"Is model spurious? {'Yes' if ecm_results['is_spurious'] else 'No'}")
    
    # Plot residuals of ECM model
    plt.figure(figsize=(12, 6))
    plt.plot(ecm_results['residuals'])
    plt.title('Residuals of the Error Correction Model (ECM)')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.grid(True)
    plt.savefig('Residuals of the Error Correction Model (ECM)')
    
    plot_residual_histogram(ecm_results['residuals'], 'Histogram of ECM Residuals')

main()
