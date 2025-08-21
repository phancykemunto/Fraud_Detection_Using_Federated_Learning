# Introduction

Fraudulent transactions are one of the biggest challenges in today’s financial world. A stolen credit card used for online shopping, a fake account transferring money across borders, or a sudden spike in unusual payments—these can cost millions of dollars in losses and erode customer trust overnight.

Now imagine global networks like Visa, MasterCard, or PayPal. Each handles millions of transactions daily. On their own, they may detect certain fraud patterns, but many fraudsters operate across multiple platforms, making it hard for any single institution to see the full picture.

Here’s the problem: while these organizations could benefit from sharing data to strengthen fraud detection, privacy laws and security concerns prevent them from exchanging sensitive customer information.

This is where Federated Learning (FL) comes in. With FL, each institution trains a model locally on its own transaction data, then securely shares only the model updates (not raw data) with a central server. The server aggregates these updates to improve a global fraud detection model.

This way, Visa, MasterCard, PayPal, and even banks can collaboratively detect fraud patterns that none of them could catch alone—without ever compromising customer privacy.

This project explores how federated learning can be applied to financial fraud detection by simulating multiple institutions training models on their own data and combining their knowledge into a stronger global model. By doing so, it demonstrates the potential of FL to enhance fraud detection accuracy while safeguarding sensitive financial information.

# Problem Statement

The key challenge addressed in this project is detecting fraudulent financial transactions across multiple institutions without compromising data privacy. Traditional centralized approaches require pooling sensitive data in one location, which poses serious risks in terms of security, compliance, and user trust. At the same time, isolated models at individual institutions often lack the broader fraud patterns needed for effective detection. This project seeks to solve this dilemma by leveraging federated learning, which enables collaborative model training while ensuring that raw transaction data never leaves its source.

## Dataset Decsription

**Transaction_id**: Serves as a unique identifier for each transaction. While not directly predictive, it ensures data integrity and helps track individual transactions during analysis.

**Transaction_type**: Different types of transactions (e.g., TRANSFER, CASH_OUT) have different risk levels. For example, cash-out transactions may be more frequently associated with fraud compared to regular deposits.

**Transaction_amount**: Unusually high or low amounts compared to a user’s normal behavior can indicate potential fraud. Sudden spikes are a red flag for monitoring.

**Timestamp**: The time and date of a transaction can reveal suspicious patterns, such as transactions at unusual hours or on holidays.

**Source_account_type & destination_account_type**: Certain account types may be more prone to fraud. Tracking both the sender and receiver helps identify irregular fund flows.

**Transaction_mode**: Online, ATM, or POS transactions carry different risk profiles. Online transactions may be more vulnerable to hacking, whereas ATM withdrawals can indicate physical card misuse.

**Balance_before & balance_after**: Sudden changes in balance, especially large withdrawals or transfers, are strong indicators of abnormal activity.

**Currency**: Detects transactions in unexpected or foreign currencies that may indicate international fraud attempts.

**Transaction_frequency**: Accounts with an unusual spike in activity compared to their typical behavior can be flagged as suspicious.

**Average_transaction_value**: Helps identify transactions that deviate significantly from the user’s normal spending pattern.

**Device_type**: The type of device used can highlight potential fraud; for instance, a login from a new or unusual device may signal a compromised account.

**Ip_location_region**: Transactions originating from an unusual geographic location, especially far from the user’s typical location, can indicate fraudulent activity.

**Fraud_indicator**: The target variable for supervised learning. It labels each transaction as fraudulent (1) or legitimate (0), allowing models to learn patterns associated with fraud.
