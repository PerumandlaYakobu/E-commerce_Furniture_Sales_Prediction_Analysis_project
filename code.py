# [cite_start]Step 1: Import Required Libraries [cite: 280, 281]
import pandas as pd #
import numpy as np #
import matplotlib.pyplot as plt #
import seaborn as sns #
from sklearn.feature_extraction.text import TfidfVectorizer #
from sklearn.model_selection import train_test_split #
from sklearn.linear_model import LinearRegression #
from sklearn.ensemble import Randomত্যাForestRegressor #
from sklearn.metrics import mean_squared_error, r2_score #
from sklearn.preprocessing import LabelEncoder #

# --- Start of Project Code ---

# [cite_start]Step 2: Load the Dataset [cite: 279, 282]
# Assume the dataset 'ecommerce_furniture_dataset.csv' is in the same directory as your Python script.
# If your file is named differently (e.g., 'ecommerce_furniture_dataset_2024.csv' as seen in the document output),
# please update the filename below.
file_path = 'ecommerce_furniture_dataset.csv' # Default name from document text
# If your file is 'ecommerce_furniture_dataset_2024.csv' as shown in Kaggle output, uncomment the line below:
# file_path = 'ecommerce_furniture_dataset_2024.csv'

try:
    df = pd.read_csv(file_path) #
    print(f"Dataset loaded successfully from: {file_path}")
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please ensure the dataset file is in the same directory as this script.")
    print("If your file is named differently (e.g., 'ecommerce_furniture_dataset_2024.csv'), please update the 'file_path' variable in the code.")
    exit() # Exit the script if the dataset isn't found

# [cite_start]View the first few rows of the dataset [cite: 284, 285]
print("\nFirst 5 rows of the dataset:")
print(df.head()) #

# [cite_start]Check the shape of the dataset [cite: 385]
print("\nShape of the dataset:", df.shape) #

# [cite_start]Step 3: Data Preprocessing [cite: 286, 287]
print("\n--- Data Preprocessing ---")

# [cite_start]Check for missing values [cite: 288, 289]
print("\nMissing values before cleaning:")
print(df.isnull().sum()) #

# Handle missing values in 'originalPrice' and 'tagText' as per the document's implied steps.
# [cite_start]The document's example output shows originalPrice having 1513 missing values[cite: 385].
# [cite_start]The document's example output also shows tagText having 3 missing values[cite: 385].
# [cite_start]The original document's preprocessing section first checks for nulls, then `df = df.dropna()`[cite: 290].
# This would drop all rows with any missing values. Let's apply that for consistency.
initial_rows = df.shape[0]
[cite_start]df.dropna(inplace=True) # [cite: 290]
print(f"\nDropped {initial_rows - df.shape[0]} rows with missing values.")
print("Shape after dropping nulls:", df.shape)

# Convert 'price' column to numeric. It might be read as object due to '$'.
if 'price' in df.columns:
    # [cite_start]Remove '$' and ',' from 'price' column and convert to float [cite: 446, 447]
    df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float) #
    print("\n'price' column cleaned and converted to float.")
else:
    print("\nWarning: 'price' column not found. Skipping price cleaning.")

# If 'originalPrice' exists and still contains '$', clean it too (though it might have been dropped by dropna)
if 'originalPrice' in df.columns:
    df['originalPrice'] = df['originalPrice'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)
    print("'originalPrice' column cleaned and converted to float.")


# Convert tagText into a categorical feature if needed, then simplify it as per document's example.
# [cite_start]Document shows `df['tagText'] = df['tagText'].astype('category').cat.codes` [cite: 292]
# However, later in the document, it simplifies tagText values and then uses LabelEncoder.
# We will follow the document's explicit simplification first, then LabelEncoder for final numerical representation.
if 'tagText' in df.columns:
    # [cite_start]Replace all values except 'Free shipping' and '+Shipping: $5.09' with 'others' [cite: 414, 415, 416, 417]
    df['tagText'] = df['tagText'].apply(lambda x: x if x in ['Free shipping', '+Shipping: $5.09'] else 'others') #
    print("\n'tagText' values simplified to 'Free shipping', '+Shipping: $5.09', or 'others'.")
    [cite_start]print(df['tagText'].value_counts()) # [cite: 418, 419, 420, 421, 422, 423, 424, 425, 426]

    # [cite_start]Convert 'tagText' to numerical using LabelEncoder [cite: 561, 562, 563, 564]
    le = LabelEncoder() #
    df['tagText'] = le.fit_transform(df['tagText']) #
    print("\n'tagText' column encoded using LabelEncoder.")
    [cite_start]print(df['tagText'].value_counts()) # [cite: 585, 586, 587, 588, 589, 590, 591, 592, 593, 594]
else:
    print("\nWarning: 'tagText' column not found. Skipping tagText processing.")


# [cite_start]Checking for data types and conversions if necessary [cite: 293]
print("\nData types after preprocessing:")
print(df.info()) #

# [cite_start]Step 4: Exploratory Data Analysis (EDA) [cite: 294]
print("\n--- Exploratory Data Analysis (EDA) ---")

# [cite_start]Distribution of 'sold' values [cite: 298, 299]
if 'sold' in df.columns:
    plt.figure(figsize=(10, 6))
    [cite_start]sns.histplot(df['sold'], kde=True) # Corrected to histplot from distplot (deprecated warning) [cite: 299]
    [cite_start]plt.title('Distribution of Furniture Items Sold') # [cite: 300]
    plt.xlabel('Units Sold')
    plt.ylabel('Count')
    plt.show() #

    # [cite_start]Plot the relationship between price and sold [cite: 512, 513]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='price', y='sold', data=df) #
    plt.title('Relationship Between Price and Items Sold')
    plt.xlabel('Price')
    plt.ylabel('Units Sold')
    plt.show()

    # [cite_start]Plot the relationship between originalPrice, price and sold [cite: 301]
    # This might fail if originalPrice was entirely dropped due to NaNs earlier.
    # Check for both columns before plotting.
    if 'originalPrice' in df.columns and 'price' in df.columns and 'sold' in df.columns:
        [cite_start]sns.pairplot(df, vars=['originalPrice', 'price', 'sold'], kind='scatter') # [cite: 301]
        [cite_start]plt.suptitle('Relationship Between Price, Original Price, and Items Sold', y=1.02) # [cite: 302, 303]
        plt.show() #
    else:
        print("\nSkipping pairplot: 'originalPrice', 'price', or 'sold' column not available for analysis.")
else:
    print("\nWarning: 'sold' column not found. Skipping EDA for 'sold' distribution and relationships.")

# [cite_start]Count plot for tagText (after simplification) [cite: 430]
if 'tagText' in df.columns:
    plt.figure(figsize=(8, 5))
    [cite_start]sns.countplot(x='tagText', data=df) # [cite: 430]
    plt.title('Distribution of tagText Categories')
    # Map back to original labels for readability if desired, or explain encoding.
    plt.xticks(ticks=[0, 1, 2], labels=['+Shipping: $5.09', 'Free shipping', 'others']) # Adjust based on LabelEncoder output if different
    plt.xlabel('Tag Text')
    plt.ylabel('Count')
    plt.show()
else:
    print("\nSkipping tagText countplot: 'tagText' column not available.")


# [cite_start]Step 5: Feature Engineering [cite: 305]
print("\n--- Feature Engineering ---")

# [cite_start]Create a new feature: percentage discount [cite: 309, 310, 311]
# Ensure 'originalPrice' is available and not zero to avoid division by zero.
if 'originalPrice' in df.columns and 'price' in df.columns:
    # Replace 0s in originalPrice with NaN temporarily to prevent ZeroDivisionError, then fill with mean/median or drop
    df['originalPrice_safe'] = df['originalPrice'].replace(0, np.nan)
    # Fill NaN originalPrice_safe values with the mean of non-zero original prices to allow discount calculation
    # Or, a more robust approach might be to set discount to 0 if originalPrice is 0 or NaN.
    # For simplicity, let's fill NaNs to allow calculation as per the original problem's intent for a discount feature.
    df['originalPrice_safe'].fillna(df['originalPrice_safe'].mean(), inplace=True) # Fill with mean for calculation

    df['discount_percentage'] = ((df['originalPrice_safe'] - df['price']) / df['originalPrice_safe']) * 100 #
    # Handle cases where discount might be negative (price > originalPrice) or very high due to data anomalies.
    # Clip values to a reasonable range, e.g., 0 to 100
    df['discount_percentage'] = df['discount_percentage'].clip(lower=0, upper=100)
    print("\n'discount_percentage' feature created.")
    df.drop('originalPrice_safe', axis=1, inplace=True) # Drop the temporary column
else:
    print("\nWarning: 'originalPrice' or 'price' column not found. Skipping 'discount_percentage' feature engineering.")


# [cite_start]Convert productTitle into a numeric feature using TF-IDF Vectorizer [cite: 312]
if 'productTitle' in df.columns:
    [cite_start]tfidf = TfidfVectorizer(max_features=100) # [cite: 313]
    # Ensure productTitle is string type before fitting TF-IDF
    [cite_start]productTitle_tfidf = tfidf.fit_transform(df['productTitle'].astype(str)) # [cite: 314]

    # [cite_start]Convert to DataFrame and concatenate to original df [cite: 315, 316, 317]
    productTitle_tfidf_df = pd.DataFrame(productTitle_tfidf.toarray(), columns=tfidf.get_feature_names_out()) #
    
    # Reset index of both dataframes before concatenation to avoid alignment issues if rows were dropped
    df.reset_index(drop=True, inplace=True)
    productTitle_tfidf_df.reset_index(drop=True, inplace=True)

    [cite_start]df = pd.concat([df, productTitle_tfidf_df], axis=1) # [cite: 318]

    # [cite_start]Drop original productTitle as it's now encoded [cite: 319, 320]
    df = df.drop('productTitle', axis=1) #
    print("'productTitle' converted to numerical features using TF-IDF and original column dropped.")
else:
    print("\nWarning: 'productTitle' column not found. Skipping TF-IDF feature engineering.")

# Ensure 'sold' and 'price' (and 'discount_percentage' if created) are numeric.
# This should largely be handled by earlier cleaning, but a final check is good.
# Also ensure all TF-IDF columns are numeric (they should be after .toarray()).
for col in ['sold', 'price', 'discount_percentage']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill any NaNs that might have resulted from coercion, e.g., with median/mean or 0
        df[col].fillna(df[col].median() if col != 'sold' else 0, inplace=True) # Fill sold with 0 if NaN

print("\nFinal DataFrame head after feature engineering:")
print(df.head())
print("\nFinal DataFrame info after feature engineering:")
print(df.info())


# [cite_start]Step 6: Model Selection & Training [cite: 321]
print("\n--- Model Selection & Training ---")

# Define features (X) and target (y)
# [cite_start]Drop 'sold' as target variable [cite: 327]
# Also drop 'originalPrice' if it was kept and 'discount_percentage' is now the primary price-related feature,
# or keep both if desired for the model. The original code only drops 'productTitle'.
# We need to make sure all columns in X are numeric.
[cite_start]X = df.drop('sold', axis=1) # [cite: 327]
[cite_start]y = df['sold'] # [cite: 328]

# Drop any non-numeric columns that might still be in X after previous steps if they weren't transformed.
# This is a safeguard if, for example, 'originalPrice' was missing and not cleaned earlier, or if tagText wasn't encoded.
X = X.select_dtypes(include=np.number)
print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# [cite_start]Train-test split (80% train, 20% test) [cite: 329]
# Use stratify=y if your 'sold' target is categorical/discrete and you want balanced classes.
# For regression, it's often not strictly necessary unless you have very skewed distributions of target values.
[cite_start]X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # [cite: 330]
print("Dataset split into training and testing sets.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# [cite_start]Initialize models [cite: 331]
[cite_start]lr_model = LinearRegression() # [cite: 332]
[cite_start]rf_model = RandomعاForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # [cite: 333]
# Added n_jobs=-1 for Random Forest to use all available CPU cores, making it faster.

# [cite_start]Train models [cite: 334]
print("\nTraining Linear Regression model...")
[cite_start]lr_model.fit(X_train, y_train) # [cite: 335]
print("Linear Regression model trained.")

print("Training Random Forest Regressor model...")
[cite_start]rf_model.fit(X_train, y_train) # [cite: 336]
print("Random Forest Regressor model trained.")


# [cite_start]Step 7: Model Evaluation [cite: 337, 338]
print("\n--- Model Evaluation ---")

# [cite_start]Predict with Linear Regression [cite: 339]
y_pred_lr = lr_model.predict(X_test) #
mse_lr = mean_squared_error(y_test, y_pred_lr) #
r2_lr = r2_score(y_test, y_pred_lr) #

# [cite_start]Predict with Random Forest [cite: 341, 342]
y_pred_rf = rf_model.predict(X_test) #
mse_rf = mean_squared_error(y_test, y_pred_rf) #
r2_rf = r2_score(y_test, y_pred_rf) #

# [cite_start]Print model evaluation results [cite: 344, 345]
print(f'Linear Regression MSE: {mse_lr:.2f}, R2: {r2_lr:.2f}') # Formatted output for readability
print(f'Random Forest MSE: {mse_rf:.2f}, R2: {r2_rf:.2f}') #

# [cite_start]Step 8: Conclusion (as described in the document) [cite: 346, 347, 348]
print("\n--- Conclusion ---")
[cite_start]print("After evaluating the models, we can conclude which model performed better and further tune hyperparameters if needed.") # [cite: 347]
[cite_start]print("Random Forest tends to perform better on complex datasets with high variance, while Linear Regression might work well if relationships are linear.") # [cite: 348]
print("\nOutput Summary:") #
[cite_start]print(f"1. Linear Regression Model: MSE = {mse_lr:.2f}, R2 = {r2_lr:.2f}") # [cite: 350]
[cite_start]print(f"2. Random Forest Model: MSE = {mse_rf:.2f}, R2 = {r2_rf:.2f}") # [cite: 351]

print("\nProject execution complete. Check your plot windows for visualizations.")