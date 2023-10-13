#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# In[2]:


Retail_data = pd.read_csv(r'C:\Users\Olamilekan .A. David\OneDrive\Documents\Data Analytics\Quantium\Customer Analysis\QVI_data.csv')
Retail_data.head()


# In[3]:


# Summary statistics for numerical columns for better understanding
print(Retail_data.describe())


# In[4]:


# Frequency counts for categorical columns
for column in Retail_data.select_dtypes(include=['object']).columns:
    print(Retail_data[column].value_counts())


# In[5]:


# setting my plot themes
import matplotlib.pyplot as plt
plt.style.use('ggplot') 


# In[6]:


# adding new month ID with the format yyyymm
Retail_data['DATE'] = pd.to_datetime(Retail_data['DATE'])
Retail_data['YEARMONTH'] = Retail_data['DATE'].dt.to_period('M')


# In[7]:


Retail_data.head()


# In[8]:


# filter stores that are present throughout the pre-trial period
pre_trial_data = Retail_data[Retail_data['DATE'] < '2019-02-01']
stores_present_pre_trial = pre_trial_data['STORE_NBR'].unique()


# In[9]:


stores_present_pre_trial


# In[10]:


# Calculate metrics of Interest over time for each staore
# Monthly overall sales revenue
monthly_revenue = pre_trial_data.groupby(['STORE_NBR', 'YEARMONTH'])['TOT_SALES'].sum()
monthly_revenue


# In[11]:


# Monthly number of customers
monthly_customers = pre_trial_data.groupby(['STORE_NBR', 'YEARMONTH'])['LYLTY_CARD_NBR'].nunique()
monthly_customers


# In[12]:


# Monthly number of transactions per customer
monthly_transactions_per_customer = pre_trial_data.groupby(['STORE_NBR', 'YEARMONTH']).size() / monthly_customers
monthly_transactions_per_customer


# In[13]:


# Group the data by store number and year-month
grouped_data = Retail_data.groupby(['STORE_NBR', 'YEARMONTH'])

# Total sales
totSales = grouped_data['TOT_SALES'].sum()

# Number of customers
nCustomers = grouped_data['LYLTY_CARD_NBR'].nunique()

# Transactions per customer
nTxnPerCust = grouped_data.size() / nCustomers

# Chips per transaction
nChipsPerTxn = grouped_data['PROD_QTY'].sum() / grouped_data.size()

# Average price per unit
avgPricePerUnit = totSales / grouped_data['PROD_QTY'].sum()

# Combine all measures into a single DataFrame
measureOverTime = pd.DataFrame({
    'totSales': totSales,
    'nCustomers': nCustomers,
    'nTxnPerCust': nTxnPerCust,
    'nChipsPerTxn': nChipsPerTxn,
    'avgPricePerUnit': avgPricePerUnit
}).reset_index()


# In[14]:


measureOverTime


# In[15]:


# Filter to stores with full observation periods
storesWithFullObs = measureOverTime.groupby('STORE_NBR').filter(lambda x: len(x) == 12)['STORE_NBR'].unique()

# Filter to the pre-trial period
preTrialMeasures = measureOverTime[(measureOverTime['YEARMONTH'] < '2019-02') & (measureOverTime['STORE_NBR'].isin(storesWithFullObs))]


# In[16]:


def calculate_correlation(inputTable, metricCol, storeComparison):
    # Initialize an empty DataFrame to store the correlation measures
    calcCorrTable = pd.DataFrame(columns=['Store1', 'Store2', 'corr_measure'])
    
    # Get the unique store numbers
    storeNumbers = inputTable['STORE_NBR'].unique()
    
    # Loop through each store number
    for i in storeNumbers:
        # Get the data for the trial store and the current store
        store1_data = inputTable[inputTable['STORE_NBR'] == storeComparison][metricCol]
        store2_data = inputTable[inputTable['STORE_NBR'] == i][metricCol]
        
        # If the trial store and current store have different numbers of data points, skip to the next iteration
        if len(store1_data) != len(store2_data):
            continue
        
        # Calculate the correlation measure between the trial store and current store
        corr_measure = np.corrcoef(store1_data, store2_data)[0, 1]
        
        # Create a DataFrame with the trial store number, current store number, and correlation measure
        calculatedMeasure = pd.DataFrame({"Store1": [storeComparison], "Store2": [i], "corr_measure": [corr_measure]})
        
        # Append the calculated measure to the correlation table
        calcCorrTable = pd.concat([calcCorrTable, calculatedMeasure], ignore_index=True)
    
    # Return the correlation table
    return calcCorrTable


# In[17]:


# calling the calcCorrTable
output = calculate_correlation(measureOverTime, 'totSales', 77)

# To view the output
print(output)


# In[18]:


def calculate_magnitude_distance(inputTable, metricCol, storeComparison):
    # Initialize an empty DataFrame to store the magnitude distances
    calcDistTable = pd.DataFrame(columns=['Store1', 'Store2', 'YEARMONTH', 'measure'])
    
    # Get the unique store numbers
    storeNumbers = inputTable['STORE_NBR'].unique()
    
    # Loop through each store number
    for i in storeNumbers:
        # Get the data for the trial store and the current store
        store1_data = inputTable[inputTable['STORE_NBR'] == storeComparison]
        store2_data = inputTable[inputTable['STORE_NBR'] == i]

        # Merge store1_data and store2_data on 'YEARMONTH'
        merged_data = pd.merge(store1_data, store2_data, on='YEARMONTH', suffixes=('_store1', '_store2'))

        # Calculate the absolute difference measure between the trial store and current store
        measure = abs(merged_data[metricCol + '_store1'].values - merged_data[metricCol + '_store2'].values)

        # Create a DataFrame with the trial store number, current store number, year-month, and measure
        calculatedMeasure = pd.DataFrame({
            "Store1": [storeComparison] * len(measure),
            "Store2": [i] * len(measure),
            "YEARMONTH": merged_data['YEARMONTH'].values,
            "measure": measure
    })

        
        # Append the calculated measure to the magnitude distance table
        calcDistTable = pd.concat([calcDistTable, calculatedMeasure], ignore_index=True)
    
    # Return the magnitude distance table
    return calcDistTable


# In[19]:


# Using 'Retail_data' is your DataFrame, 'TOT_SALES' is the metric column, and 77 is the trial store number
output = calculate_magnitude_distance(measureOverTime, 'totSales', 77)

# To view the output
print(output)


# In[20]:


def calculate_magnitude_distance(inputTable, metricCol, storeComparison):
    # Initialize an empty DataFrame to store the magnitude distances
    calcDistTable = pd.DataFrame(columns=['Store1', 'Store2', 'YEARMONTH', 'measure'])
    
    # Get the unique store numbers
    storeNumbers = inputTable['STORE_NBR'].unique()
    
    # Loop through each store number
    for i in storeNumbers:
        # Get the data for the trial store and the current store
        store1_data = inputTable[inputTable['STORE_NBR'] == storeComparison]
        store2_data = inputTable[inputTable['STORE_NBR'] == i]

        # Merge store1_data and store2_data on 'YEARMONTH'
        merged_data = pd.merge(store1_data, store2_data, on='YEARMONTH', suffixes=('_store1', '_store2'))

        # Calculate the absolute difference measure between the trial store and current store
        measure = abs(merged_data[metricCol + '_store1'].values - merged_data[metricCol + '_store2'].values)

        # Create a DataFrame with the trial store number, current store number, year-month, and measure
        calculatedMeasure = pd.DataFrame({
            "Store1": [storeComparison] * len(measure),
            "Store2": [i] * len(measure),
            "YEARMONTH": merged_data['YEARMONTH'].values,
            "measure": measure
    })
        
        # Append the calculated measure to the magnitude distance table
        calcDistTable = pd.concat([calcDistTable, calculatedMeasure], ignore_index=True)
    
    # Calculate min and max of measure for each Store1 and YEARMONTH combination
    minMaxDist = calcDistTable.groupby(['Store1', 'YEARMONTH'])['measure'].agg(['min', 'max']).reset_index()
    
    # Merge minMaxDist with calcDistTable
    distTable = pd.merge(calcDistTable, minMaxDist, on=['Store1', 'YEARMONTH'])
    
    # Standardise the magnitude distance so that it ranges from 0 to 1
    distTable['magnitudeMeasure'] = 1 - (distTable['measure'] - distTable['min']) / (distTable['max'] - distTable['min'])
    
    # Calculate mean of magnitudeMeasure for each Store1 and Store2 combination
    finalDistTable = distTable.groupby(['Store1', 'Store2'])['magnitudeMeasure'].mean().reset_index()
    
    # Return the final distance table
    return finalDistTable


# In[21]:


#getting finalDistTable
output = calculate_magnitude_distance(measureOverTime, 'totSales', 77)

# To view the output
print(output)


# In[22]:


# Having 'Retail_data' DataFrame, 'TOT_SALES' is the metric column, and 77 is the trial store number
corrTable = calculate_correlation(measureOverTime, 'totSales', 77)
distTable = calculate_magnitude_distance(measureOverTime, 'totSales', 77)

# Merge corrTable and distTable
combinedTable = pd.merge(corrTable, distTable, on=['Store1', 'Store2'], suffixes=('_corr', '_dist'))

# Calculate the combined score
combinedTable['score'] = 0.5 * combinedTable['corr_measure'] + 0.5 * combinedTable['magnitudeMeasure']

# To view the output
print(combinedTable)


# In[23]:


# Using 'measureOverTime' DataFrame, 'totSales' and 'nCustomers' are the metric columns, and 77 is the trial store number
score_nSales = calculate_correlation(measureOverTime, 'totSales', 77)
score_nCustomers = calculate_correlation(measureOverTime, 'nCustomers', 77)

# Merge score_nSales and score_nCustomers
score_Control = pd.merge(score_nSales, score_nCustomers, on=['Store1', 'Store2'], suffixes=('_nSales', '_nCustomers'))

# Calculate the combined score
score_Control['finalControlScore'] = 0.5 * score_Control['corr_measure_nSales'] + 0.5 * score_Control['corr_measure_nCustomers']

# To view the output
print(score_Control)


# In[25]:


#selecting control_store
# Having 'score_Control' DataFrame and 77 is the trial store number
control_store = score_Control[(score_Control['Store1'] == 77) & (score_Control['Store2'] != 77)].nlargest(1, 'finalControlScore')['Store2'].values[0]

# To view the control store
print(control_store)


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt

# Define your trial and control store numbers
trial_store = 77
control_store = 41

# Create a new column 'Store_type' in the DataFrame based on the store number
measureOverTime['Store_type'] = measureOverTime['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Convert 'YEARMONTH' to a datetime object and store it in a new column 'TransactionMonth'
measureOverTime['TransactionMonth'] = measureOverTime['YEARMONTH'].dt.to_timestamp()

# Filter rows where 'YEARMONTH' is before March 2019
pastSales = measureOverTime[measureOverTime['TransactionMonth'] < pd.to_datetime('2019-03')]

# Group by 'TransactionMonth' and 'Store_type', and calculate the mean of 'totSales'
pastSales = pastSales.groupby(['TransactionMonth', 'Store_type'])['totSales'].mean().reset_index()

# Create a new figure
plt.figure(figsize=(12, 6))

# For each store type, plot total sales over time
for store_type in ['Trial', 'Control', 'Other stores']:
    data = pastSales[pastSales['Store_type'] == store_type]
    plt.plot(data['TransactionMonth'], data['totSales'], label=store_type)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Total sales')
plt.title('Total sales by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()



# In[32]:


# Define your trial and control store numbers
trial_store = 77
control_store = 41

# Create a new column 'Store_type' in the DataFrame based on the store number
measureOverTime['Store_type'] = measureOverTime['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Convert 'YEARMONTH' to a datetime object and store it in a new column 'TransactionMonth'
measureOverTime['TransactionMonth'] = measureOverTime['YEARMONTH'].dt.to_timestamp()

# Filter rows where 'YEARMONTH' is before March 2019
pastCustomers = measureOverTime[measureOverTime['TransactionMonth'] < pd.to_datetime('2019-03')]

# Group by 'TransactionMonth' and 'Store_type', and calculate the mean of 'nCustomers'
pastCustomers = pastCustomers.groupby(['TransactionMonth', 'Store_type'])['nCustomers'].mean().reset_index()

# Create a new figure
plt.figure(figsize=(12, 6))

# For each store type, plot total customers over time
for store_type in ['Trial', 'Control', 'Other stores']:
    data = pastCustomers[pastCustomers['Store_type'] == store_type]
    plt.plot(data['TransactionMonth'], data['nCustomers'], label=store_type)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Number of customers')
plt.title('Number of customers by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()



# In[33]:


# Calculate the scaling factor for control sales
scalingFactorForControlSales = preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == trial_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'totSales'].sum() / preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == control_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'totSales'].sum()

# Apply the scaling factor to control store's sales
measureOverTimeSales = measureOverTime.copy()
scaledControlSales = measureOverTimeSales.loc[measureOverTimeSales['STORE_NBR'] == control_store].copy()
scaledControlSales['controlSales'] = scaledControlSales['totSales'] * scalingFactorForControlSales


# In[34]:


# Merge the trial store's sales data with the scaled control store's sales data
percentageDiff = pd.merge(measureOverTimeSales.loc[measureOverTimeSales['STORE_NBR'] == trial_store], scaledControlSales, on='YEARMONTH')

# Calculate the percentage difference
percentageDiff['percentageDiff'] = abs((percentageDiff['totSales_x'] - percentageDiff['controlSales']) / percentageDiff['controlSales'])


# In[35]:


# Display the DataFrame
display(percentageDiff)


# In[36]:


import numpy as np
from scipy import stats

# Calculate the standard deviation of the percentage difference during the pre-trial period
stdDev = percentageDiff.loc[percentageDiff['YEARMONTH'] < '2019-02', 'percentageDiff'].std()

# Define the degrees of freedom
degreesOfFreedom = 7

# Calculate the t-values for the trial months
percentageDiff['tValue'] = (percentageDiff['percentageDiff'] - 0) / stdDev

# Filter rows for the trial period
trial_period_tvalues = percentageDiff.loc[(percentageDiff['YEARMONTH'] >= '2019-02') & (percentageDiff['YEARMONTH'] <= '2019-04')]

# Print the t-values for the trial period
print(trial_period_tvalues)

# Calculate the 95th percentile of the t-distribution
critical_t_value = stats.t.ppf(1-0.05, df=degreesOfFreedom)

# Print the critical t-value
print("Critical t-value: ", critical_t_value)


# In[39]:


import pandas as pd
import matplotlib.pyplot as plt

# Define your trial and control store numbers
trial_store = 77
control_store = 41

# Create a new column 'Store_type' in the DataFrame based on the store number
measureOverTimeSales['Store_type'] = measureOverTimeSales['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Convert 'YEARMONTH' to a datetime object and store it in a new column 'TransactionMonth'
measureOverTimeSales['TransactionMonth'] = measureOverTimeSales['YEARMONTH'].dt.to_timestamp()

# Filter rows for trial and control stores
pastSales = measureOverTimeSales.loc[measureOverTimeSales['Store_type'].isin(['Trial', 'Control'])]

# Calculate control store's 95th percentile sales
pastSales_Controls95 = pastSales.loc[pastSales['Store_type'] == "Control"].copy()
pastSales_Controls95['totSales'] = pastSales_Controls95['totSales'] * (1 + stdDev * 2)
pastSales_Controls95['Store_type'] = "Control 95th % confidence interval"

# Calculate control store's 5th percentile sales
pastSales_Controls5 = pastSales.loc[pastSales['Store_type'] == "Control"].copy()
pastSales_Controls5['totSales'] = pastSales_Controls5['totSales'] * (1 - stdDev * 2)
pastSales_Controls5['Store_type'] = "Control 5th % confidence interval"

# Combine the data
trialAssessment = pd.concat([pastSales, pastSales_Controls95, pastSales_Controls5])

# Create a new figure
plt.figure(figsize=(12, 6))

# For each store type, plot total sales over time
for store_type in ['Trial', 'Control', 'Control 95th % confidence interval', 'Control 5th % confidence interval']:
    data = trialAssessment[trialAssessment['Store_type'] == store_type]
    plt.plot(data['TransactionMonth'], data['totSales'], label=store_type)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Total sales')
plt.title('Total sales by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[41]:


# Calculate the scaling factor for control customers
scalingFactorForControlCust = preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == trial_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'nCustomers'].sum() / preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == control_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'nCustomers'].sum()

# Apply the scaling factor to control store's customer counts
measureOverTimeCusts = measureOverTime.copy()
scaledControlCustomers = measureOverTimeCusts.loc[measureOverTimeCusts['STORE_NBR'] == control_store].copy()
scaledControlCustomers['controlCustomers'] = scaledControlCustomers['nCustomers'] * scalingFactorForControlCust

# Create a new column 'Store_type' in the DataFrame based on the store number
scaledControlCustomers['Store_type'] = scaledControlCustomers['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Calculate the percentage difference between scaled control customers and trial customers
percentageDiff = pd.merge(measureOverTimeCusts.loc[measureOverTimeCusts['STORE_NBR'] == trial_store], scaledControlCustomers, on='YEARMONTH')
percentageDiff['percentageDiff'] = abs((percentageDiff['nCustomers_x'] - percentageDiff['controlCustomers']) / percentageDiff['controlCustomers'])


# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a new figure
fig, ax = plt.subplots(figsize=(12, 6))

# For each store type, plot total customers over time
for store_type in ['Trial', 'Control', 'Control 95th % confidence interval', 'Control 5th % confidence interval']:
    data = trialAssessment[trialAssessment['Store_type'] == store_type]
    ax.plot(data['TransactionMonth'], data['nCustomers'], label=store_type)

# Highlight the trial period
trial_start = pd.to_datetime('2019-02')
trial_end = pd.to_datetime('2019-04')
rect = patches.Rectangle((trial_start, 0), trial_end-trial_start, np.inf, linewidth=0, edgecolor='none', facecolor='gray', alpha=0.2)
ax.add_patch(rect)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Number of customers')
plt.title('Number of customers by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[44]:


# store 86, all functions to recalculated

Retail_data.head(3)


# In[46]:


#calculating metrics as earlier done

# Group the data by store number and year-month
grouped_data = Retail_data.groupby(['STORE_NBR', 'YEARMONTH'])

# Total sales
totSales = grouped_data['TOT_SALES'].sum()

# Number of customers
nCustomers = grouped_data['LYLTY_CARD_NBR'].nunique()

# Transactions per customer
nTxnPerCust = grouped_data.size() / nCustomers

# Chips per transaction
nChipsPerTxn = grouped_data['PROD_QTY'].sum() / grouped_data.size()

# Average price per unit
avgPricePerUnit = totSales / grouped_data['PROD_QTY'].sum()

# Combine all measures into a single DataFrame
measureOverTime = pd.DataFrame({
    'totSales': totSales,
    'nCustomers': nCustomers,
    'nTxnPerCust': nTxnPerCust,
    'nChipsPerTxn': nChipsPerTxn,
    'avgPricePerUnit': avgPricePerUnit
}).reset_index()


# In[47]:


# Filter to stores with full observation periods
storesWithFullObs = measureOverTime.groupby('STORE_NBR').filter(lambda x: len(x) == 12)['STORE_NBR'].unique()

# Filter to the pre-trial period
preTrialMeasures = measureOverTime[(measureOverTime['YEARMONTH'] < '2019-02') & (measureOverTime['STORE_NBR'].isin(storesWithFullObs))]


# In[48]:


def calculate_correlation(inputTable, metricCol, storeComparison):
    # Initialize an empty DataFrame to store the correlation measures
    calcCorrTable = pd.DataFrame(columns=['Store1', 'Store2', 'corr_measure'])
    
    # Get the unique store numbers
    storeNumbers = inputTable['STORE_NBR'].unique()
    
    # Loop through each store number
    for i in storeNumbers:
        # Get the data for the trial store and the current store
        store1_data = inputTable[inputTable['STORE_NBR'] == storeComparison][metricCol]
        store2_data = inputTable[inputTable['STORE_NBR'] == i][metricCol]
        
        # If the trial store and current store have different numbers of data points, skip to the next iteration
        if len(store1_data) != len(store2_data):
            continue
        
        # Calculate the correlation measure between the trial store and current store
        corr_measure = np.corrcoef(store1_data, store2_data)[0, 1]
        
        # Create a DataFrame with the trial store number, current store number, and correlation measure
        calculatedMeasure = pd.DataFrame({"Store1": [storeComparison], "Store2": [i], "corr_measure": [corr_measure]})
        
        # Append the calculated measure to the correlation table
        calcCorrTable = pd.concat([calcCorrTable, calculatedMeasure], ignore_index=True)
    
    # Return the correlation table
    return calcCorrTable


# In[49]:


# calling the calcCorrTable
output = calculate_correlation(measureOverTime, 'totSales', 86)

# To view the output
print(output)


# In[50]:


def calculate_magnitude_distance(inputTable, metricCol, storeComparison):
    # Initialize an empty DataFrame to store the magnitude distances
    calcDistTable = pd.DataFrame(columns=['Store1', 'Store2', 'YEARMONTH', 'measure'])
    
    # Get the unique store numbers
    storeNumbers = inputTable['STORE_NBR'].unique()
    
    # Loop through each store number
    for i in storeNumbers:
        # Get the data for the trial store and the current store
        store1_data = inputTable[inputTable['STORE_NBR'] == storeComparison]
        store2_data = inputTable[inputTable['STORE_NBR'] == i]

        # Merge store1_data and store2_data on 'YEARMONTH'
        merged_data = pd.merge(store1_data, store2_data, on='YEARMONTH', suffixes=('_store1', '_store2'))

        # Calculate the absolute difference measure between the trial store and current store
        measure = abs(merged_data[metricCol + '_store1'].values - merged_data[metricCol + '_store2'].values)

        # Create a DataFrame with the trial store number, current store number, year-month, and measure
        calculatedMeasure = pd.DataFrame({
            "Store1": [storeComparison] * len(measure),
            "Store2": [i] * len(measure),
            "YEARMONTH": merged_data['YEARMONTH'].values,
            "measure": measure
    })

        
        # Append the calculated measure to the magnitude distance table
        calcDistTable = pd.concat([calcDistTable, calculatedMeasure], ignore_index=True)
    
    # Return the magnitude distance table
    return calcDistTable


# In[51]:


# Using 'Retail_data' is your DataFrame, 'TOT_SALES' is the metric column, and 77 is the trial store number
output = calculate_magnitude_distance(measureOverTime, 'totSales', 86)

# To view the output
print(output)


# In[52]:


def calculate_magnitude_distance(inputTable, metricCol, storeComparison):
    # Initialize an empty DataFrame to store the magnitude distances
    calcDistTable = pd.DataFrame(columns=['Store1', 'Store2', 'YEARMONTH', 'measure'])
    
    # Get the unique store numbers
    storeNumbers = inputTable['STORE_NBR'].unique()
    
    # Loop through each store number
    for i in storeNumbers:
        # Get the data for the trial store and the current store
        store1_data = inputTable[inputTable['STORE_NBR'] == storeComparison]
        store2_data = inputTable[inputTable['STORE_NBR'] == i]

        # Merge store1_data and store2_data on 'YEARMONTH'
        merged_data = pd.merge(store1_data, store2_data, on='YEARMONTH', suffixes=('_store1', '_store2'))

        # Calculate the absolute difference measure between the trial store and current store
        measure = abs(merged_data[metricCol + '_store1'].values - merged_data[metricCol + '_store2'].values)

        # Create a DataFrame with the trial store number, current store number, year-month, and measure
        calculatedMeasure = pd.DataFrame({
            "Store1": [storeComparison] * len(measure),
            "Store2": [i] * len(measure),
            "YEARMONTH": merged_data['YEARMONTH'].values,
            "measure": measure
    })
        
        # Append the calculated measure to the magnitude distance table
        calcDistTable = pd.concat([calcDistTable, calculatedMeasure], ignore_index=True)
    
    # Calculate min and max of measure for each Store1 and YEARMONTH combination
    minMaxDist = calcDistTable.groupby(['Store1', 'YEARMONTH'])['measure'].agg(['min', 'max']).reset_index()
    
    # Merge minMaxDist with calcDistTable
    distTable = pd.merge(calcDistTable, minMaxDist, on=['Store1', 'YEARMONTH'])
    
    # Standardise the magnitude distance so that it ranges from 0 to 1
    distTable['magnitudeMeasure'] = 1 - (distTable['measure'] - distTable['min']) / (distTable['max'] - distTable['min'])
    
    # Calculate mean of magnitudeMeasure for each Store1 and Store2 combination
    finalDistTable = distTable.groupby(['Store1', 'Store2'])['magnitudeMeasure'].mean().reset_index()
    
    # Return the final distance table
    return finalDistTable


# In[53]:


#getting finalDistTable
output = calculate_magnitude_distance(measureOverTime, 'totSales', 86)

# To view the output
print(output)


# In[54]:


# Having 'Retail_data' DataFrame, 'TOT_SALES' is the metric column, and 86 is the trial store number
corrTable = calculate_correlation(measureOverTime, 'totSales', 86)
distTable = calculate_magnitude_distance(measureOverTime, 'totSales', 86)

# Merge corrTable and distTable
combinedTable = pd.merge(corrTable, distTable, on=['Store1', 'Store2'], suffixes=('_corr', '_dist'))

# Calculate the combined score
combinedTable['score'] = 0.5 * combinedTable['corr_measure'] + 0.5 * combinedTable['magnitudeMeasure']

# To view the output
print(combinedTable)


# In[55]:


# Using 'measureOverTime' DataFrame, 'totSales' and 'nCustomers' are the metric columns, and 86 is the trial store number
score_nSales = calculate_correlation(measureOverTime, 'totSales', 86)
score_nCustomers = calculate_correlation(measureOverTime, 'nCustomers', 86)

# Merge score_nSales and score_nCustomers
score_Control = pd.merge(score_nSales, score_nCustomers, on=['Store1', 'Store2'], suffixes=('_nSales', '_nCustomers'))

# Calculate the combined score
score_Control['finalControlScore'] = 0.5 * score_Control['corr_measure_nSales'] + 0.5 * score_Control['corr_measure_nCustomers']

# To view the output
print(score_Control)


# In[56]:


#selecting control_store
# Having 'score_Control' DataFrame and 86 is the trial store number
control_store = score_Control[(score_Control['Store1'] == 86) & (score_Control['Store2'] != 86)].nlargest(1, 'finalControlScore')['Store2'].values[0]

# To view the control store
print(control_store)


# In[62]:


import pandas as pd
import matplotlib.pyplot as plt

# Define your trial and control store numbers
trial_store = 86
control_store = 229

# Create a new column 'Store_type' in the DataFrame based on the store number
measureOverTime['Store_type'] = measureOverTime['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Convert 'YEARMONTH' to a datetime object and store it in a new column 'TransactionMonth'
measureOverTime['TransactionMonth'] = measureOverTime['YEARMONTH'].dt.to_timestamp()

# Filter rows where 'YEARMONTH' is before March 2019
pastSales = measureOverTime[measureOverTime['TransactionMonth'] < pd.to_datetime('2019-03')]

# Group by 'TransactionMonth' and 'Store_type', and calculate the mean of 'totSales'
pastSales = pastSales.groupby(['TransactionMonth', 'Store_type'])['totSales'].mean().reset_index()

# Create a new figure
plt.figure(figsize=(12, 6))

# For each store type, plot total sales over time
for store_type in ['Trial', 'Control', 'Other stores']:
    data = pastSales[pastSales['Store_type'] == store_type]
    plt.plot(data['TransactionMonth'], data['totSales'], label=store_type)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Total sales')
plt.title('Total sales by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[63]:


# Define your trial and control store numbers
trial_store = 86
control_store = 229

# Create a new column 'Store_type' in the DataFrame based on the store number
measureOverTime['Store_type'] = measureOverTime['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Convert 'YEARMONTH' to a datetime object and store it in a new column 'TransactionMonth'
measureOverTime['TransactionMonth'] = measureOverTime['YEARMONTH'].dt.to_timestamp()

# Filter rows where 'YEARMONTH' is before March 2019
pastCustomers = measureOverTime[measureOverTime['TransactionMonth'] < pd.to_datetime('2019-03')]

# Group by 'TransactionMonth' and 'Store_type', and calculate the mean of 'nCustomers'
pastCustomers = pastCustomers.groupby(['TransactionMonth', 'Store_type'])['nCustomers'].mean().reset_index()

# Create a new figure
plt.figure(figsize=(12, 6))

# For each store type, plot total customers over time
for store_type in ['Trial', 'Control', 'Other stores']:
    data = pastCustomers[pastCustomers['Store_type'] == store_type]
    plt.plot(data['TransactionMonth'], data['nCustomers'], label=store_type)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Number of customers')
plt.title('Number of customers by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[64]:


# Calculate the scaling factor for control sales
scalingFactorForControlSales = preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == trial_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'totSales'].sum() / preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == control_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'totSales'].sum()

# Apply the scaling factor to control store's sales
measureOverTimeSales = measureOverTime.copy()
scaledControlSales = measureOverTimeSales.loc[measureOverTimeSales['STORE_NBR'] == control_store].copy()
scaledControlSales['controlSales'] = scaledControlSales['totSales'] * scalingFactorForControlSales


# In[65]:


# Merge the trial store's sales data with the scaled control store's sales data
percentageDiff = pd.merge(measureOverTimeSales.loc[measureOverTimeSales['STORE_NBR'] == trial_store], scaledControlSales, on='YEARMONTH')

# Calculate the percentage difference
percentageDiff['percentageDiff'] = abs((percentageDiff['totSales_x'] - percentageDiff['controlSales']) / percentageDiff['controlSales'])


# In[66]:


# Display the DataFrame
display(percentageDiff)


# In[67]:


import numpy as np
from scipy import stats

# Calculate the standard deviation of the percentage difference during the pre-trial period
stdDev = percentageDiff.loc[percentageDiff['YEARMONTH'] < '2019-02', 'percentageDiff'].std()

# Define the degrees of freedom
degreesOfFreedom = 7

# Calculate the t-values for the trial months
percentageDiff['tValue'] = (percentageDiff['percentageDiff'] - 0) / stdDev

# Filter rows for the trial period
trial_period_tvalues = percentageDiff.loc[(percentageDiff['YEARMONTH'] >= '2019-02') & (percentageDiff['YEARMONTH'] <= '2019-04')]

# Print the t-values for the trial period
print(trial_period_tvalues)

# Calculate the 95th percentile of the t-distribution
critical_t_value = stats.t.ppf(1-0.05, df=degreesOfFreedom)

# Print the critical t-value
print("Critical t-value: ", critical_t_value)


# In[70]:


import pandas as pd
import matplotlib.pyplot as plt

# Define your trial and control store numbers
trial_store = 86
control_store = 229

# Create a new column 'Store_type' in the DataFrame based on the store number
measureOverTimeSales['Store_type'] = measureOverTimeSales['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Convert 'YEARMONTH' to a datetime object and store it in a new column 'TransactionMonth'
measureOverTimeSales['TransactionMonth'] = measureOverTimeSales['YEARMONTH'].dt.to_timestamp()

# Filter rows for trial and control stores
pastSales = measureOverTimeSales.loc[measureOverTimeSales['Store_type'].isin(['Trial', 'Control'])]

# Calculate control store's 95th percentile sales
pastSales_Controls95 = pastSales.loc[pastSales['Store_type'] == "Control"].copy()
pastSales_Controls95['totSales'] = pastSales_Controls95['totSales'] * (1 + stdDev * 2)
pastSales_Controls95['Store_type'] = "Control 95th % confidence interval"

# Calculate control store's 5th percentile sales
pastSales_Controls5 = pastSales.loc[pastSales['Store_type'] == "Control"].copy()
pastSales_Controls5['totSales'] = pastSales_Controls5['totSales'] * (1 - stdDev * 2)
pastSales_Controls5['Store_type'] = "Control 5th % confidence interval"

# Combine the data
trialAssessment = pd.concat([pastSales, pastSales_Controls95, pastSales_Controls5])

# Create a new figure
plt.figure(figsize=(12, 6))

# For each store type, plot total sales over time
for store_type in ['Trial', 'Control', 'Control 95th % confidence interval', 'Control 5th % confidence interval']:
    data = trialAssessment[trialAssessment['Store_type'] == store_type]
    plt.plot(data['TransactionMonth'], data['totSales'], label=store_type)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Total sales')
plt.title('Total sales by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[71]:


# Calculate the scaling factor for control customers
scalingFactorForControlCust = preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == trial_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'nCustomers'].sum() / preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == control_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'nCustomers'].sum()

# Apply the scaling factor to control store's customer counts
measureOverTimeCusts = measureOverTime.copy()
scaledControlCustomers = measureOverTimeCusts.loc[measureOverTimeCusts['STORE_NBR'] == control_store].copy()
scaledControlCustomers['controlCustomers'] = scaledControlCustomers['nCustomers'] * scalingFactorForControlCust

# Create a new column 'Store_type' in the DataFrame based on the store number
scaledControlCustomers['Store_type'] = scaledControlCustomers['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Calculate the percentage difference between scaled control customers and trial customers
percentageDiff = pd.merge(measureOverTimeCusts.loc[measureOverTimeCusts['STORE_NBR'] == trial_store], scaledControlCustomers, on='YEARMONTH')
percentageDiff['percentageDiff'] = abs((percentageDiff['nCustomers_x'] - percentageDiff['controlCustomers']) / percentageDiff['controlCustomers'])


# In[72]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a new figure
fig, ax = plt.subplots(figsize=(12, 6))

# For each store type, plot total customers over time
for store_type in ['Trial', 'Control', 'Control 95th % confidence interval', 'Control 5th % confidence interval']:
    data = trialAssessment[trialAssessment['Store_type'] == store_type]
    ax.plot(data['TransactionMonth'], data['nCustomers'], label=store_type)

# Highlight the trial period
trial_start = pd.to_datetime('2019-02')
trial_end = pd.to_datetime('2019-04')
rect = patches.Rectangle((trial_start, 0), trial_end-trial_start, np.inf, linewidth=0, edgecolor='none', facecolor='gray', alpha=0.2)
ax.add_patch(rect)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Number of customers')
plt.title('Number of customers by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[73]:


# Trial store 88

#calculating metrics as earlier done

# Group the data by store number and year-month
grouped_data = Retail_data.groupby(['STORE_NBR', 'YEARMONTH'])

# Total sales
totSales = grouped_data['TOT_SALES'].sum()

# Number of customers
nCustomers = grouped_data['LYLTY_CARD_NBR'].nunique()

# Transactions per customer
nTxnPerCust = grouped_data.size() / nCustomers

# Chips per transaction
nChipsPerTxn = grouped_data['PROD_QTY'].sum() / grouped_data.size()

# Average price per unit
avgPricePerUnit = totSales / grouped_data['PROD_QTY'].sum()

# Combine all measures into a single DataFrame
measureOverTime = pd.DataFrame({
    'totSales': totSales,
    'nCustomers': nCustomers,
    'nTxnPerCust': nTxnPerCust,
    'nChipsPerTxn': nChipsPerTxn,
    'avgPricePerUnit': avgPricePerUnit
}).reset_index()


# In[74]:


# Filter to stores with full observation periods
storesWithFullObs = measureOverTime.groupby('STORE_NBR').filter(lambda x: len(x) == 12)['STORE_NBR'].unique()

# Filter to the pre-trial period
preTrialMeasures = measureOverTime[(measureOverTime['YEARMONTH'] < '2019-02') & (measureOverTime['STORE_NBR'].isin(storesWithFullObs))]


# In[75]:


def calculate_correlation(inputTable, metricCol, storeComparison):
    # Initialize an empty DataFrame to store the correlation measures
    calcCorrTable = pd.DataFrame(columns=['Store1', 'Store2', 'corr_measure'])
    
    # Get the unique store numbers
    storeNumbers = inputTable['STORE_NBR'].unique()
    
    # Loop through each store number
    for i in storeNumbers:
        # Get the data for the trial store and the current store
        store1_data = inputTable[inputTable['STORE_NBR'] == storeComparison][metricCol]
        store2_data = inputTable[inputTable['STORE_NBR'] == i][metricCol]
        
        # If the trial store and current store have different numbers of data points, skip to the next iteration
        if len(store1_data) != len(store2_data):
            continue
        
        # Calculate the correlation measure between the trial store and current store
        corr_measure = np.corrcoef(store1_data, store2_data)[0, 1]
        
        # Create a DataFrame with the trial store number, current store number, and correlation measure
        calculatedMeasure = pd.DataFrame({"Store1": [storeComparison], "Store2": [i], "corr_measure": [corr_measure]})
        
        # Append the calculated measure to the correlation table
        calcCorrTable = pd.concat([calcCorrTable, calculatedMeasure], ignore_index=True)
    
    # Return the correlation table
    return calcCorrTable


# In[76]:


# calling the calcCorrTable
output = calculate_correlation(measureOverTime, 'totSales', 88)

# To view the output
print(output)


# In[77]:


def calculate_magnitude_distance(inputTable, metricCol, storeComparison):
    # Initialize an empty DataFrame to store the magnitude distances
    calcDistTable = pd.DataFrame(columns=['Store1', 'Store2', 'YEARMONTH', 'measure'])
    
    # Get the unique store numbers
    storeNumbers = inputTable['STORE_NBR'].unique()
    
    # Loop through each store number
    for i in storeNumbers:
        # Get the data for the trial store and the current store
        store1_data = inputTable[inputTable['STORE_NBR'] == storeComparison]
        store2_data = inputTable[inputTable['STORE_NBR'] == i]

        # Merge store1_data and store2_data on 'YEARMONTH'
        merged_data = pd.merge(store1_data, store2_data, on='YEARMONTH', suffixes=('_store1', '_store2'))

        # Calculate the absolute difference measure between the trial store and current store
        measure = abs(merged_data[metricCol + '_store1'].values - merged_data[metricCol + '_store2'].values)

        # Create a DataFrame with the trial store number, current store number, year-month, and measure
        calculatedMeasure = pd.DataFrame({
            "Store1": [storeComparison] * len(measure),
            "Store2": [i] * len(measure),
            "YEARMONTH": merged_data['YEARMONTH'].values,
            "measure": measure
    })

        
        # Append the calculated measure to the magnitude distance table
        calcDistTable = pd.concat([calcDistTable, calculatedMeasure], ignore_index=True)
    
    # Return the magnitude distance table
    return calcDistTable


# In[78]:


# Using 'Retail_data' is your DataFrame, 'TOT_SALES' is the metric column, and 77 is the trial store number
output = calculate_magnitude_distance(measureOverTime, 'totSales', 88)

# To view the output
print(output)


# In[79]:


def calculate_magnitude_distance(inputTable, metricCol, storeComparison):
    # Initialize an empty DataFrame to store the magnitude distances
    calcDistTable = pd.DataFrame(columns=['Store1', 'Store2', 'YEARMONTH', 'measure'])
    
    # Get the unique store numbers
    storeNumbers = inputTable['STORE_NBR'].unique()
    
    # Loop through each store number
    for i in storeNumbers:
        # Get the data for the trial store and the current store
        store1_data = inputTable[inputTable['STORE_NBR'] == storeComparison]
        store2_data = inputTable[inputTable['STORE_NBR'] == i]

        # Merge store1_data and store2_data on 'YEARMONTH'
        merged_data = pd.merge(store1_data, store2_data, on='YEARMONTH', suffixes=('_store1', '_store2'))

        # Calculate the absolute difference measure between the trial store and current store
        measure = abs(merged_data[metricCol + '_store1'].values - merged_data[metricCol + '_store2'].values)

        # Create a DataFrame with the trial store number, current store number, year-month, and measure
        calculatedMeasure = pd.DataFrame({
            "Store1": [storeComparison] * len(measure),
            "Store2": [i] * len(measure),
            "YEARMONTH": merged_data['YEARMONTH'].values,
            "measure": measure
    })
        
        # Append the calculated measure to the magnitude distance table
        calcDistTable = pd.concat([calcDistTable, calculatedMeasure], ignore_index=True)
    
    # Calculate min and max of measure for each Store1 and YEARMONTH combination
    minMaxDist = calcDistTable.groupby(['Store1', 'YEARMONTH'])['measure'].agg(['min', 'max']).reset_index()
    
    # Merge minMaxDist with calcDistTable
    distTable = pd.merge(calcDistTable, minMaxDist, on=['Store1', 'YEARMONTH'])
    
    # Standardise the magnitude distance so that it ranges from 0 to 1
    distTable['magnitudeMeasure'] = 1 - (distTable['measure'] - distTable['min']) / (distTable['max'] - distTable['min'])
    
    # Calculate mean of magnitudeMeasure for each Store1 and Store2 combination
    finalDistTable = distTable.groupby(['Store1', 'Store2'])['magnitudeMeasure'].mean().reset_index()
    
    # Return the final distance table
    return finalDistTable


# In[80]:


#getting finalDistTable
output = calculate_magnitude_distance(measureOverTime, 'totSales', 88)

# To view the output
print(output)


# In[81]:


# Having 'Retail_data' DataFrame, 'TOT_SALES' is the metric column, and 86 is the trial store number
corrTable = calculate_correlation(measureOverTime, 'totSales', 88)
distTable = calculate_magnitude_distance(measureOverTime, 'totSales', 88)

# Merge corrTable and distTable
combinedTable = pd.merge(corrTable, distTable, on=['Store1', 'Store2'], suffixes=('_corr', '_dist'))

# Calculate the combined score
combinedTable['score'] = 0.5 * combinedTable['corr_measure'] + 0.5 * combinedTable['magnitudeMeasure']

# To view the output
print(combinedTable)


# In[82]:


# Using 'measureOverTime' DataFrame, 'totSales' and 'nCustomers' are the metric columns, and 86 is the trial store number
score_nSales = calculate_correlation(measureOverTime, 'totSales', 88)
score_nCustomers = calculate_correlation(measureOverTime, 'nCustomers', 88)

# Merge score_nSales and score_nCustomers
score_Control = pd.merge(score_nSales, score_nCustomers, on=['Store1', 'Store2'], suffixes=('_nSales', '_nCustomers'))

# Calculate the combined score
score_Control['finalControlScore'] = 0.5 * score_Control['corr_measure_nSales'] + 0.5 * score_Control['corr_measure_nCustomers']

# To view the output
print(score_Control)


# In[83]:


#selecting control_store
# Having 'score_Control' DataFrame and 86 is the trial store number
control_store = score_Control[(score_Control['Store1'] == 88) & (score_Control['Store2'] != 88)].nlargest(1, 'finalControlScore')['Store2'].values[0]

# To view the control store
print(control_store)


# In[86]:


import pandas as pd
import matplotlib.pyplot as plt

# Define your trial and control store numbers
trial_store = 88
control_store = 178

# Create a new column 'Store_type' in the DataFrame based on the store number
measureOverTime['Store_type'] = measureOverTime['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Convert 'YEARMONTH' to a datetime object and store it in a new column 'TransactionMonth'
measureOverTime['TransactionMonth'] = measureOverTime['YEARMONTH'].dt.to_timestamp()

# Filter rows where 'YEARMONTH' is before March 2019
pastSales = measureOverTime[measureOverTime['TransactionMonth'] < pd.to_datetime('2019-03')]

# Group by 'TransactionMonth' and 'Store_type', and calculate the mean of 'totSales'
pastSales = pastSales.groupby(['TransactionMonth', 'Store_type'])['totSales'].mean().reset_index()

# Create a new figure
plt.figure(figsize=(12, 6))

# For each store type, plot total sales over time
for store_type in ['Trial', 'Control', 'Other stores']:
    data = pastSales[pastSales['Store_type'] == store_type]
    plt.plot(data['TransactionMonth'], data['totSales'], label=store_type)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Total sales')
plt.title('Total sales by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[87]:


# Define your trial and control store numbers
trial_store = 88
control_store = 178

# Create a new column 'Store_type' in the DataFrame based on the store number
measureOverTime['Store_type'] = measureOverTime['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Convert 'YEARMONTH' to a datetime object and store it in a new column 'TransactionMonth'
measureOverTime['TransactionMonth'] = measureOverTime['YEARMONTH'].dt.to_timestamp()

# Filter rows where 'YEARMONTH' is before March 2019
pastCustomers = measureOverTime[measureOverTime['TransactionMonth'] < pd.to_datetime('2019-03')]

# Group by 'TransactionMonth' and 'Store_type', and calculate the mean of 'nCustomers'
pastCustomers = pastCustomers.groupby(['TransactionMonth', 'Store_type'])['nCustomers'].mean().reset_index()

# Create a new figure
plt.figure(figsize=(12, 6))

# For each store type, plot total customers over time
for store_type in ['Trial', 'Control', 'Other stores']:
    data = pastCustomers[pastCustomers['Store_type'] == store_type]
    plt.plot(data['TransactionMonth'], data['nCustomers'], label=store_type)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Number of customers')
plt.title('Number of customers by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[88]:


# Calculate the scaling factor for control sales
scalingFactorForControlSales = preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == trial_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'totSales'].sum() / preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == control_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'totSales'].sum()

# Apply the scaling factor to control store's sales
measureOverTimeSales = measureOverTime.copy()
scaledControlSales = measureOverTimeSales.loc[measureOverTimeSales['STORE_NBR'] == control_store].copy()
scaledControlSales['controlSales'] = scaledControlSales['totSales'] * scalingFactorForControlSales


# In[89]:


# Merge the trial store's sales data with the scaled control store's sales data
percentageDiff = pd.merge(measureOverTimeSales.loc[measureOverTimeSales['STORE_NBR'] == trial_store], scaledControlSales, on='YEARMONTH')

# Calculate the percentage difference
percentageDiff['percentageDiff'] = abs((percentageDiff['totSales_x'] - percentageDiff['controlSales']) / percentageDiff['controlSales'])


# In[90]:


# Display the DataFrame
display(percentageDiff)


# In[91]:


import numpy as np
from scipy import stats

# Calculate the standard deviation of the percentage difference during the pre-trial period
stdDev = percentageDiff.loc[percentageDiff['YEARMONTH'] < '2019-02', 'percentageDiff'].std()

# Define the degrees of freedom
degreesOfFreedom = 7

# Calculate the t-values for the trial months
percentageDiff['tValue'] = (percentageDiff['percentageDiff'] - 0) / stdDev

# Filter rows for the trial period
trial_period_tvalues = percentageDiff.loc[(percentageDiff['YEARMONTH'] >= '2019-02') & (percentageDiff['YEARMONTH'] <= '2019-04')]

# Print the t-values for the trial period
print(trial_period_tvalues)

# Calculate the 95th percentile of the t-distribution
critical_t_value = stats.t.ppf(1-0.05, df=degreesOfFreedom)

# Print the critical t-value
print("Critical t-value: ", critical_t_value)


# In[92]:


import pandas as pd
import matplotlib.pyplot as plt

# Define your trial and control store numbers
trial_store = 88
control_store = 178

# Create a new column 'Store_type' in the DataFrame based on the store number
measureOverTimeSales['Store_type'] = measureOverTimeSales['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Convert 'YEARMONTH' to a datetime object and store it in a new column 'TransactionMonth'
measureOverTimeSales['TransactionMonth'] = measureOverTimeSales['YEARMONTH'].dt.to_timestamp()

# Filter rows for trial and control stores
pastSales = measureOverTimeSales.loc[measureOverTimeSales['Store_type'].isin(['Trial', 'Control'])]

# Calculate control store's 95th percentile sales
pastSales_Controls95 = pastSales.loc[pastSales['Store_type'] == "Control"].copy()
pastSales_Controls95['totSales'] = pastSales_Controls95['totSales'] * (1 + stdDev * 2)
pastSales_Controls95['Store_type'] = "Control 95th % confidence interval"

# Calculate control store's 5th percentile sales
pastSales_Controls5 = pastSales.loc[pastSales['Store_type'] == "Control"].copy()
pastSales_Controls5['totSales'] = pastSales_Controls5['totSales'] * (1 - stdDev * 2)
pastSales_Controls5['Store_type'] = "Control 5th % confidence interval"

# Combine the data
trialAssessment = pd.concat([pastSales, pastSales_Controls95, pastSales_Controls5])

# Create a new figure
plt.figure(figsize=(12, 6))

# For each store type, plot total sales over time
for store_type in ['Trial', 'Control', 'Control 95th % confidence interval', 'Control 5th % confidence interval']:
    data = trialAssessment[trialAssessment['Store_type'] == store_type]
    plt.plot(data['TransactionMonth'], data['totSales'], label=store_type)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Total sales')
plt.title('Total sales by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[93]:


# Calculate the scaling factor for control customers
scalingFactorForControlCust = preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == trial_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'nCustomers'].sum() / preTrialMeasures.loc[(preTrialMeasures['STORE_NBR'] == control_store) & (preTrialMeasures['YEARMONTH'] < '2019-02'), 'nCustomers'].sum()

# Apply the scaling factor to control store's customer counts
measureOverTimeCusts = measureOverTime.copy()
scaledControlCustomers = measureOverTimeCusts.loc[measureOverTimeCusts['STORE_NBR'] == control_store].copy()
scaledControlCustomers['controlCustomers'] = scaledControlCustomers['nCustomers'] * scalingFactorForControlCust

# Create a new column 'Store_type' in the DataFrame based on the store number
scaledControlCustomers['Store_type'] = scaledControlCustomers['STORE_NBR'].apply(lambda x: 'Trial' if x == trial_store else ('Control' if x == control_store else 'Other stores'))

# Calculate the percentage difference between scaled control customers and trial customers
percentageDiff = pd.merge(measureOverTimeCusts.loc[measureOverTimeCusts['STORE_NBR'] == trial_store], scaledControlCustomers, on='YEARMONTH')
percentageDiff['percentageDiff'] = abs((percentageDiff['nCustomers_x'] - percentageDiff['controlCustomers']) / percentageDiff['controlCustomers'])


# In[94]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a new figure
fig, ax = plt.subplots(figsize=(12, 6))

# For each store type, plot total customers over time
for store_type in ['Trial', 'Control', 'Control 95th % confidence interval', 'Control 5th % confidence interval']:
    data = trialAssessment[trialAssessment['Store_type'] == store_type]
    ax.plot(data['TransactionMonth'], data['nCustomers'], label=store_type)

# Highlight the trial period
trial_start = pd.to_datetime('2019-02')
trial_end = pd.to_datetime('2019-04')
rect = patches.Rectangle((trial_start, 0), trial_end-trial_start, np.inf, linewidth=0, edgecolor='none', facecolor='gray', alpha=0.2)
ax.add_patch(rect)

# Set labels and title for the plot
plt.xlabel('Month of operation')
plt.ylabel('Number of customers')
plt.title('Number of customers by month')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[ ]:




