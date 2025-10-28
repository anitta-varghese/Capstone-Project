import pandas as pd
import numpy as np
from tabulate import tabulate  # Add this

df1 = pd.read_csv('C:/project/Inputt.csv', encoding='latin1')
df = df1[['Instrument', 'delta', 'Bucket_No', 'RW', 'Corr']]

#Net Sensitivity Aggregation
aggregated_df = df.groupby(['Instrument', 'Bucket_No', 'RW', 'Corr'], as_index=False)['delta'].sum()
print("\n📌 Net Sensitivities:")
print(tabulate(aggregated_df, headers='keys', tablefmt='fancy_grid', showindex=False))

#Weighted Sensitivities
aggregated_df['WSk'] = aggregated_df['delta'] * aggregated_df['RW']
aggregated_df = aggregated_df[['Bucket_No', 'WSk', 'Corr']].sort_values(by='Bucket_No')

print("\n📌 Weighted Sensitivities (WSk = delta * RW):")
print(tabulate(aggregated_df, headers='keys', tablefmt='fancy_grid', showindex=False))

#Intra-Bucket Aggregation
def aggregate_bucket(df):
    grouped_results = {}

    for bucket, group in df.groupby('Bucket_No'):
        W = group['WSk'].values
        rho = group['Corr'].iloc[0]  # Assuming constant correlation within bucket

        matrix = np.zeros((len(W), len(W)))
        for i in range(len(W)):
            for j in range(len(W)):
                matrix[i, j] = W[i] * W[j] if i == j else rho * W[i] * W[j]
        # Variance aggregation formula  
        variance_sum = np.sum(np.diag(matrix))
        covariance_sum = np.sum(matrix) - variance_sum
        K_b = np.sqrt(variance_sum + covariance_sum)  # Final aggregated value
        grouped_results[bucket] = K_b

        df_matrix = pd.DataFrame(matrix, columns=[f"Sens_{i+1}" for i in range(len(W))],
                                 index=[f"Sens_{i+1}" for i in range(len(W))])
        print(f"\n📌 Intra-Bucket Matrix for Bucket {bucket}:")
        print(tabulate(df_matrix, headers='keys', tablefmt='fancy_grid'))

    return grouped_results

#Intra-Bucket Aggregation
aggregated_df1 = aggregate_bucket(aggregated_df)

aggregated_df1_df = pd.DataFrame(list(aggregated_df1.items()), columns=['Bucket_No', 'Aggregated_WSk'])
print("\n📌 Intra-Bucket Aggregation Results:")
print(tabulate(aggregated_df1_df, headers='keys', tablefmt='fancy_grid', showindex=False))

#Inter-Bucket Covariance Calculation
bucket_list = sorted(aggregated_df1.keys())
inter_bucket_matrix = pd.DataFrame(0, index=bucket_list, columns=bucket_list, dtype=float)

correlation_matrix = pd.DataFrame([
    [1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0, 0.45, 0.45],
    [0.15, 1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0, 0.45, 0.45],
    [0.15, 0.15, 1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0, 0.45, 0.45],
    [0.15, 0.15, 0.15, 1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0, 0.45, 0.45],
    [0.15, 0.15, 0.15, 0.15, 1, 0.15, 0.15, 0.15, 0.15, 0.15, 0, 0.45, 0.45],
    [0.15, 0.15, 0.15, 0.15, 0.15, 1, 0.15, 0.15, 0.15, 0.15, 0, 0.45, 0.45],
    [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 1, 0.15, 0.15, 0.15, 0, 0.45, 0.45],
    [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 1, 0.15, 0.15, 0, 0.45, 0.45],
    [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 1, 0.15, 0, 0.45, 0.45],
    [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 1, 0, 0.45, 0.45],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0, 1, 0.75],
    [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0, 0.75, 1]
], index=range(1, 14), columns=range(1, 14))

 # Initialize sum
for i in bucket_list:
    for j in bucket_list:
        rho_ij = correlation_matrix.loc[i, j] # Get correlation
        inter_bucket_matrix.loc[i, j] = aggregated_df1[i] * aggregated_df1[j] * rho_ij

print("\n📌 Inter-Bucket Covariance Matrix:")
print(tabulate(inter_bucket_matrix, headers='keys', tablefmt='fancy_grid'))

#Final Capital Charge
S = np.sqrt(inter_bucket_matrix.to_numpy().sum())
print(f"\n✅ Final Inter-Bucket Aggregated Value (Total Capital Charge): {S:.4f}")



