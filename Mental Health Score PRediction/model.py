import pandas as pd 

df = pd.read_csv("Mental Health Score PRediction/survey.csv")
# Create a copy
df = df.copy()

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Drop unnecessary columns
df.drop(columns=['Timestamp', 'state', 'comments'], inplace=True)

# Handle missing values
df['self_employed'] = df['self_employed'].fillna(
    df['self_employed'].mode()[0]
)

df['work_interfere'] = df['work_interfere'].fillna('Unknown')

# Clean invalid age values
df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())
# Separate features and target

X = df.drop('treatment', axis=1)
y = df['treatment']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Encode input categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

X[categorical_columns] = encoder.fit_transform(
    X[categorical_columns]
)

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Encoding completed!")

print("\nEncoded Features:")
print(X.head())

print("\nEncoded Target:")
print(y[:10])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

# Train the model
model.fit(X_train, y_train)

print("Random Forest Model Training Completed!")
