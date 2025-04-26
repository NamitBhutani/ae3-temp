import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.ardl import ARDL, ardl_select_order
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def process_wpi_variation_data(file_path):
    try:
        raw_data = pd.read_excel(file_path, header=None)
    except Exception as e:
        print(f"Error loading WPI data: {e}")
        raise
    
    combined_data = pd.DataFrame(columns=['Year', 'WPI'])
    
    current_section = None
    header_row_idx = None
    year_col_idx = None
    ac_col_idx = None
    data_rows = []
    
    for idx, row in raw_data.iterrows():
        row_str = ' '.join([str(x) for x in row if pd.notna(x)])
        
        if 'Base' in row_str and ('2011-12' in row_str or '2004-05' in row_str or '1993-94' in row_str or '1981-82' in row_str or '1970-71' in row_str):
            if current_section and header_row_idx and year_col_idx is not None and ac_col_idx is not None and data_rows:
                section_df = pd.DataFrame(data_rows, columns=['Year', 'WPI'])
                combined_data = pd.concat([combined_data, section_df])
                data_rows = []
            
            current_section = row_str
            header_row_idx = None
            year_col_idx = None
            ac_col_idx = None
            continue
        
        if 'Year' in row_str and ('AC' in row_str ):
            header_row_idx = idx
            
            for i, val in enumerate(row):
                if pd.notna(val) and 'Year' in str(val):
                    year_col_idx = i
                elif pd.notna(val) and ('AC' in str(val) or val == 2):
                    ac_col_idx = i
            continue
        
        if header_row_idx and year_col_idx is not None and ac_col_idx is not None:
            if all(isinstance(x, (int, float)) and x in range(1, 10) for x in [val for val in row if pd.notna(val)]):
                continue
                
            year_val = row[year_col_idx]
            ac_val = row[ac_col_idx]
            
            if pd.notna(year_val) and pd.notna(ac_val) and isinstance(year_val, str) and '-' in year_val:
                year = int(year_val.split('-')[0])
                data_rows.append([year, ac_val])
    
    if current_section and header_row_idx and year_col_idx is not None and ac_col_idx is not None and data_rows:
        section_df = pd.DataFrame(data_rows, columns=['Year', 'WPI'])
        combined_data = pd.concat([combined_data, section_df])
    
    combined_data = combined_data.drop_duplicates(subset='Year', keep='first')
    
    combined_data = combined_data.sort_values('Year')
    
    combined_data.set_index('Year', inplace=True)
    combined_data = combined_data.loc[1991:2017]
    
    if combined_data.empty:
        raise ValueError("No WPI data could be extracted from the file")
    return combined_data

def parse_rbi_policy_rates(file_path):
    try:
        data = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading RBI policy rates data: {e}")
        raise
    
    if 'Effective Date' in data.columns:
        policy_data = data
    else:
        raise ValueError("Data does not have expected headers")
    
    policy_data.columns = [str(col).strip() for col in policy_data.columns]
    
    date_col = [col for col in policy_data.columns if 'Effective Date' in col][0]
    bank_rate_col = [col for col in policy_data.columns if 'Bank Rate' in col][0]
    crr_col = [col for col in policy_data.columns if 'Cash Reserve' in col or 'CRR' in col][0]
    slr_col = [col for col in policy_data.columns if 'Statutory Liquidity' in col or 'SLR' in col][0]
    
    policy_data[date_col] = pd.to_datetime(policy_data[date_col], format='%d-%m-%Y', errors='coerce')
    
    policy_data['Year'] = policy_data[date_col].dt.year
    
    for col in [bank_rate_col, crr_col, slr_col]:
        policy_data[col] = pd.to_numeric(policy_data[col], errors='coerce')
    
    yearly_rates = policy_data.sort_values(date_col).groupby('Year').last()
    
    result = pd.DataFrame({
        'BANK_RATE': yearly_rates[bank_rate_col],
        'CRR': yearly_rates[crr_col],
        'SLR': yearly_rates[slr_col]
    })
    
    min_year = min(result.index.min(), 1991)
    max_year = max(result.index.max(), 2017)
    full_years = pd.Index(range(min_year, max_year + 1))
    result = result.reindex(full_years)
    
    result = result.fillna(method='ffill')
    
    result = result.fillna(method='bfill')
    
    if result.empty:
        raise ValueError("No policy rate data could be extracted")
    
    final_result = result.loc[1991:2017]
    
    return final_result

def run_regression_model(data):
    X = data[['BANK_RATE', 'CRR', 'SLR']]
    X = sm.add_constant(X)
    y = data['WPI']
    
    model = sm.OLS(y, X).fit()
    return model

def breusch_godfrey_lm_test(model, lags=13):
    bg_test = acorr_breusch_godfrey(model, nlags=lags)
    
    return {
        'lm_stat': bg_test[0],
        'p_value': bg_test[1],
        'f_stat': bg_test[2],
        'f_p_value': bg_test[3]
    }

def breusch_pagan_godfrey_test(model):
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    
    return {
        'lm_stat': bp_test[0],
        'lm_p_value': bp_test[1],
        'f_stat': bp_test[2],
        'f_p_value': bp_test[3]
    }

def jarque_bera_test(residuals):
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
        adf_level = adfuller(data[column], maxlag=9, regression='c')
        
        first_diff = data[column].diff().dropna()
        adf_diff = adfuller(first_diff, maxlag=9, regression='c')
        
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
    johansen_results = coint_johansen(data.values, det_order=0, k_ar_diff=1)
    
    trace_stats = johansen_results.lr1
    eigen_stats = johansen_results.lr2
    
    crit_vals_trace = johansen_results.cvt
    crit_vals_eigen = johansen_results.cvm
    
    trace_summary = pd.DataFrame({
        'Hypothesized No. of CE(s)': ['None', 'At most 1', 'At most 2', 'At most 3'][:data.shape[1]],
        'Eigenvalue': johansen_results.eig,
        'Trace Statistic': trace_stats,
        '0.05 Critical Value': crit_vals_trace[:, 1],
        'p-value': [stats.chi2.sf(trace_stats[i], data.shape[1] - i) for i in range(data.shape[1])]
    })
    
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
    X = sm.add_constant(data[['BANK_RATE', 'CRR', 'SLR']])
    y = data['WPI']
    
    long_run_model = sm.OLS(y, X).fit()
    
    residuals = long_run_model.resid
    
    data_diff = data.diff().dropna()
    
    data_diff.columns = ['D' + col for col in data_diff.columns]
    
    data_diff['U1_lag1'] = residuals.shift(1).loc[data_diff.index]
    
    data_diff = data_diff.dropna()
    
    X_ecm = sm.add_constant(data_diff[['DBANK_RATE', 'DCRR', 'DSLR', 'U1_lag1']])
    y_ecm = data_diff['DWPI']
    
    ecm_model = sm.OLS(y_ecm, X_ecm).fit()
    
    dw_stat = sm.stats.stattools.durbin_watson(ecm_model.resid)
    
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
    plt.xticks(list(data.index)[::2])
    plt.savefig(title)

def plot_residual_histogram(residuals, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, stat='density')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, 'k', linewidth=2)
    jb_results = jarque_bera_test(residuals)
    jb_stat = jb_results['jb_stat']
    jb_p_value = jb_results['jb_p_value']
    plt.title(f'{title}\nJB Stat: {jb_stat:.4f}, p-value: {jb_p_value:.4f}')
    plt.grid(True)
    plt.savefig(title)

def build_ardl_model(data, p=1, q=1):
    y = data['WPI']
    exog = data[['BANK_RATE', 'CRR', 'SLR']]
    ardl_mod = ARDL(endog=y, lags=p, exog=exog, exog_lags=q).fit()
    return ardl_mod


def build_ardl_model_auto(data, maxlag=4, ic='aic', trend='c'):
    y = data['WPI']
    exog = data[['BANK_RATE', 'CRR', 'SLR']]
    sel = ardl_select_order(endog=y, exog=exog, maxorder=maxlag,maxlag=maxlag, ic=ic, trend=trend)
    print(f"Exogeneous lags: {sel.dl_lags}")
    print(f"Lags included in model: {sel.ar_lags}")
    auto_mod = sel.model.fit()
    return auto_mod, sel

def check_autocorrelation_dw(model):
    dw_stat = sm.stats.stattools.durbin_watson(model.resid)
    
    return {
        'dw_statistic': dw_stat,
        'has_autocorrelation': dw_stat < 1.5 or dw_stat > 2.5
    }

def perform_heteroskedasticity_test(residuals):
    resid_squared = residuals**2
    trend = np.arange(len(residuals))
    
    X = sm.add_constant(trend)
    
    het_model = sm.OLS(resid_squared, X).fit()
    
    f_stat = het_model.fvalue
    f_p_value = het_model.f_pvalue
    
    return {
        'f_stat': f_stat,
        'f_p_value': f_p_value,
        'has_heteroskedasticity': f_p_value < 0.05
    }

def perform_diagnostic_tests_for_ardl(model):
    residuals = model.resid
    
    dw_results = check_autocorrelation_dw(model)
    
    het_results = perform_heteroskedasticity_test(residuals)
    
    jb_results = jarque_bera_test(residuals)
    
    return {
        'autocorrelation': dw_results,
        'heteroscedasticity': het_results,
        'normality': jb_results,
        'residuals': residuals
    }

def apply_hac_standard_errors(model):
    try:
        hac_results = model.get_robustcov_results(cov_type='HAC', maxlags=4)
        return hac_results
    except Exception as e:
        print(f"Error applying HAC standard errors: {e}")
        print("Direct HAC application not supported, using original model")
        return model

def generate_summary_statistics(data):
    summary_stats = data.describe()
    
    summary_stats.loc['skewness'] = data.skew()
    summary_stats.loc['kurtosis'] = data.kurtosis()
    summary_stats.loc['median'] = data.median()
    summary_stats.loc['cv'] = data.std() / data.mean()
    
    return summary_stats

def interpret_stationarity_results(stationarity_results):
    interpretation = {}
    
    for var, results in stationarity_results.items():
        level_p = results['level']['p_value']
        diff_p = results['first_difference']['p_value']
        
        if level_p < 0.05:
            interpretation[var] = {
                'stationary_at_level': True,
                'recommendation': 'Use variable in level form',
                'integration_order': 'I(0)'
            }
        elif diff_p < 0.05:
            interpretation[var] = {
                'stationary_at_level': False,
                'recommendation': 'Use first difference',
                'integration_order': 'I(1)'
            }
        else:
            interpretation[var] = {
                'stationary_at_level': False,
                'recommendation': 'Consider higher order differencing or alternative tests',
                'integration_order': 'Higher than I(1)'
            }
    
    return interpretation

def interpret_model_coefficients(model):
    interpretation = []
    
    for var_name, coef, p_val in zip(model.model.exog_names, model.params, model.pvalues):
        significance = "significant" if p_val < 0.05 else "not significant"
        direction = "positive" if coef > 0 else "negative"
        
        interp = f"{var_name}: {coef:.4f}, {direction} effect, {significance} (p-value: {p_val:.4f})"
        interpretation.append(interp)
    
    return interpretation

def forecast_out_of_sample(data, model, sel, train_size_pct=0.8):
    n = len(data)
    train_size = int(n * train_size_pct)
    
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    y_train = train_data['WPI']
    exog_train = train_data[['BANK_RATE', 'CRR', 'SLR']]
    
    try:
        train_model = ARDL(endog=y_train, lags=sel.ar_lags, 
                         exog=exog_train).fit()
    except Exception as e:
        print(f"Error fitting ARDL model on training data: {e}")
    
    exog_forecast = test_data[['BANK_RATE', 'CRR', 'SLR']]
    
    try:
        start = len(train_data)
        end = len(data) - 1
        
        forecasts = train_model.predict(
            start=start,
            end=end,
            exog_oos=exog_forecast
        )
        
        if isinstance(forecasts, pd.Series):
            forecasts_array = forecasts.values
        else:
            forecasts_array = np.array(forecasts)
            
        actual_array = test_data['WPI'].values
        
        min_len = min(len(forecasts_array), len(actual_array))
        forecasts_array = forecasts_array[:min_len]
        actual_array = actual_array[:min_len]
             
        
        return {
            'forecasts': forecasts_array,
            'actual': actual_array,
        }
        
    except Exception as e:
        print(f"Error in forecasting: {e}")
        return {
            'forecasts': None,
            'actual': test_data['WPI'].values,
            'metrics': {
                'MSE': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan
            }
        }

def forecast_future_values(model, data, periods=3):
    try:
        last_year = data.index.max()
        
        future_years = range(last_year + 1, last_year + periods + 1)
        
        future_exog = pd.DataFrame(
            [[data.loc[last_year, 'BANK_RATE'], 
              data.loc[last_year, 'CRR'], 
              data.loc[last_year, 'SLR']]] * periods,
            columns=['BANK_RATE', 'CRR', 'SLR'],
            index=future_years
        )
        
        current_end = len(data) - 1
        
        try:
            forecasts = model.predict(
                start=current_end + 1,
                end=current_end + periods,
                exog_oos=future_exog
            )
        except TypeError:
            print("Falling back to simpler prediction method")
            forecasts = []
            for i in range(periods):
                next_pred = model.forecast(steps=1, exog=future_exog.iloc[i:i+1])
                forecasts.append(next_pred[0])
            forecasts = np.array(forecasts)
        
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'WPI_Forecast': forecasts
        })
        
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['WPI'], 'b-', label='Historical')
        plt.plot(future_years, forecasts, 'r--', label='Forecast')
        plt.title('WPI Forecasts for Future Periods')
        plt.xlabel('Year')
        plt.ylabel('WPI')
        plt.legend()
        plt.grid(True)
        plt.savefig('Future WPI Forecasts')
        
        return forecast_df
    
    except Exception as e:
        print(f"Error in future forecasting: {e}")
        future_years = range(data.index.max() + 1, data.index.max() + periods + 1)
        last_wpi = data['WPI'].iloc[-1]
        dummy_forecasts = np.array([last_wpi] * periods)
        
        print("Falling back to naive forecast (last observed value)")
        forecast_df = pd.DataFrame({
            'Year': future_years,
            'WPI_Forecast': dummy_forecasts
        })
        return forecast_df

def main():    
    wpi_data = process_wpi_variation_data("HBS Table No. 230 _ Wholesale Price Index - Annual Variation.xlsx")
    rbi_data = parse_rbi_policy_rates("HBS Table No. 43 _ Major Monetary Policy Rates and Reserve Requirements - Bank Rate, LAF (Repo, Reverse Repo and MSF) Rates, CRR & SLR.xlsx")
    data = pd.concat([wpi_data, rbi_data], axis=1)
    
    print("\n=========== i. Summary Statistics of Selected Variables ===========")
    summary_stats = generate_summary_statistics(data)
    print("\nSummary Statistics:")
    print(summary_stats)

    
    plot_time_series(data, "WPI Annual Variation (%) and RBI Policy Rates (1991-2017)")
    
    print("\n=========== ii. Stationarity Analysis ===========")
    stationarity_results = check_stationarity(data)
    
    print("\nStationarity Test Results (ADF Test):")
    for var, results in stationarity_results.items():
        print(f"\nVariable: {var}")
        print(f"At Level - p-value: {results['level']['p_value']:.4f}")
        print(f"At First Difference - p-value: {results['first_difference']['p_value']:.4f}")
    
    interpret = interpret_stationarity_results(stationarity_results)
    
    print("\nStationarity Interpretation and Recommendations:")
    for var, result in interpret.items():
        print(f"{var}: {result['integration_order']}, {result['recommendation']}")
    
    johansen_results = johansen_cointegration_test(data)
    print("\nJohansen Cointegration Test Results:")
    print("\nTrace Test:")
    print(johansen_results['trace_summary'])
    print("\nEigenvalue Test:")
    print(johansen_results['eigen_summary'])
    
    print("\n=========== iii. Model Fitting and Coefficient Interpretation ===========")
    
    model = run_regression_model(data)
    print("\nRegression Model Results:")
    print(model.summary())
    
    ecm_results = build_ecm_model(data)
    print("\nError Correction Model Results:")
    print(ecm_results['model'].summary())
    
    print("\nARDL Model with Automatic Lag Selection:")
    auto_ardl_model, auto_sel = build_ardl_model_auto(data, maxlag=4, ic='aic', trend='c')
    print(auto_ardl_model.summary())
    
    print("\nARDL Model Coefficient Interpretation:")
    ardl_coef_interpretation = interpret_model_coefficients(auto_ardl_model)
    for interp in ardl_coef_interpretation:
        print(interp)
    
    print("\nModel Selection Justification:")
    print(f"ARDL Model AIC: {auto_ardl_model.aic}")
    print(f"ARDL Model BIC: {auto_ardl_model.bic}")
    print(f"Basic OLS Model AIC: {model.aic}")
    print(f"Basic OLS Model BIC: {model.bic}")
    print(f"ECM Model AIC: {ecm_results['model'].aic}")
    print(f"ECM Model BIC: {ecm_results['model'].bic}")
    
    print("\n=========== iv. Assumption Checking and Remedial Measures ===========")
    
    print("\nOLS Model Diagnostic Tests:")
    ols_diagnostic_tests = perform_diagnostic_tests(model)
    print(f"Breusch-Godfrey LM Test p-value: {ols_diagnostic_tests['autocorrelation']['p_value']:.4f}")
    print(f"Breusch-Pagan-Godfrey Test p-value: {ols_diagnostic_tests['heteroscedasticity']['lm_p_value']:.4f}")
    print(f"Jarque-Bera Test p-value: {ols_diagnostic_tests['normality']['jb_p_value']:.4f}")
    
    print("\nARDL Model Diagnostic Tests:")
    ardl_diagnostics = perform_diagnostic_tests_for_ardl(auto_ardl_model)
    print(f"Durbin-Watson statistic: {ardl_diagnostics['autocorrelation']['dw_statistic']:.4f}")
    print(f"Heteroscedasticity F-test p-value: {ardl_diagnostics['heteroscedasticity']['f_p_value']:.4f}")
    print(f"Jarque-Bera Test - Statistic: {ardl_diagnostics['normality']['jb_stat']:.4f}, p-value: {ardl_diagnostics['normality']['jb_p_value']:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(ardl_diagnostics['residuals'])
    plt.title('Residuals of the ARDL Model')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.grid(True)
    plt.savefig('Residuals of the ARDL Model')
    
    plot_residual_histogram(ardl_diagnostics['residuals'], 'Histogram of ARDL Residuals')
    
    if ardl_diagnostics['autocorrelation']['has_autocorrelation']:
        print("\nAutocorrelation detected in ARDL model - Applying HAC standard errors")
        robust_ardl_model = apply_hac_standard_errors(auto_ardl_model)
        print("\nARDL Model with HAC Standard Errors:")
        print(robust_ardl_model.summary())
    
    print("\n=========== v. Model Modification and Conclusion ===========")
    
    final_model = auto_ardl_model
    final_sel = auto_sel
    
    if ardl_diagnostics['autocorrelation']['has_autocorrelation'] or ardl_diagnostics['heteroscedasticity']['has_heteroskedasticity']:
        print("\nTrying ARDL with different lag specifications:")
        
        bic_ardl_model, bic_sel = build_ardl_model_auto(data, maxlag=4, ic='bic', trend='c')
        print("\nARDL Model with BIC Selection:")
        print(f"Selected lags: {bic_sel.ar_lags}, {bic_sel.dl_lags}")
        
        higher_lag_model, higher_lag_sel = build_ardl_model_auto(data, maxlag=6, ic='aic', trend='c')
        print("\nARDL Model with Higher Max Lag:")
        print(f"AIC value: {higher_lag_model.aic}")
        print(f"Selected lags: {higher_lag_sel.ar_lags}, {higher_lag_sel.dl_lags}")
        
        if bic_ardl_model.bic < auto_ardl_model.bic:
            print("\nSelected BIC-based model due to better parsimony")
            final_model = bic_ardl_model
            final_sel = bic_sel
        elif higher_lag_model.aic < auto_ardl_model.aic:
            print("\nSelected higher-lag model due to better fit")
            final_model = higher_lag_model
            final_sel = higher_lag_sel
        else:
            print("\nRetained original ARDL model")
    else:
        print("\nNo model modification needed")
    
    print("\n=========== vi. Out-of-Sample Forecasting ===========")
    
    forecast_results = forecast_out_of_sample(data, final_model, final_sel)
    
    future_forecasts = forecast_future_values(final_model, data, periods=3)
    
    print("\nFuture WPI Forecasts:")
    print(future_forecasts)

if __name__ == "__main__":
    main()

