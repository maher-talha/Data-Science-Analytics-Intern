# ============================================================
# Task 3: Customer Churn Prediction (Bank Customers)
# DevelopersHub Corporation - Data Science Internship
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── 1. Create Churn Modelling Dataset (synthetic, realistic) ─
np.random.seed(42)
n = 1000

geography = np.random.choice(['France', 'Germany', 'Spain'], n, p=[0.5, 0.25, 0.25])
gender = np.random.choice(['Male', 'Female'], n)
age = np.random.randint(18, 70, n)
credit_score = np.random.randint(350, 850, n)
tenure = np.random.randint(0, 10, n)
balance = np.random.uniform(0, 250000, n)
num_of_products = np.random.choice([1, 2, 3, 4], n, p=[0.5, 0.4, 0.08, 0.02])
has_cr_card = np.random.choice([0, 1], n, p=[0.3, 0.7])
is_active_member = np.random.choice([0, 1], n, p=[0.4, 0.6])
estimated_salary = np.random.uniform(10000, 200000, n)

# Churn logic: higher age, Germany, inactive -> more churn
churn_prob = (
    0.05 +
    (age > 45).astype(float) * 0.15 +
    (geography == 'Germany').astype(float) * 0.10 +
    (is_active_member == 0).astype(float) * 0.10 +
    (num_of_products >= 3).astype(float) * 0.15
)
churn_prob = np.clip(churn_prob, 0, 1)
exited = (np.random.rand(n) < churn_prob).astype(int)

df = pd.DataFrame({
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
    'Exited': exited
})

# ── 2. Dataset Inspection ────────────────────────────────────
print("=" * 55)
print("DATASET SHAPE:", df.shape)
print("=" * 55)
print("\nFIRST 5 ROWS:")
print(df.head())
print("\nMISSING VALUES:")
print(df.isnull().sum())
print("\nCHURN DISTRIBUTION:")
print(df['Exited'].value_counts())
print(f"\nChurn Rate: {df['Exited'].mean()*100:.2f}%")

# ── 3. Label Encoding ────────────────────────────────────────
le = LabelEncoder()
df['Geography_enc'] = le.fit_transform(df['Geography'])
df['Gender_enc'] = le.fit_transform(df['Gender'])
print("\nLabel Encoding done for Geography and Gender.")

# ── 4. EDA Visualizations ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Task 3: Customer Churn EDA - Bank Customers",
             fontsize=15, fontweight='bold')

# Churn Distribution
axes[0,0].pie(df['Exited'].value_counts(), labels=['Retained', 'Churned'],
              autopct='%1.1f%%', colors=['steelblue', 'tomato'],
              startangle=90, explode=[0, 0.05])
axes[0,0].set_title('Churn Distribution')

# Age vs Churn
axes[0,1].hist(df[df['Exited']==0]['Age'], bins=25, alpha=0.6, label='Retained', color='steelblue')
axes[0,1].hist(df[df['Exited']==1]['Age'], bins=25, alpha=0.6, label='Churned', color='tomato')
axes[0,1].set_xlabel('Age')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Age Distribution by Churn')
axes[0,1].legend()

# Geography vs Churn
geo_churn = df.groupby('Geography')['Exited'].mean().reset_index()
axes[0,2].bar(geo_churn['Geography'], geo_churn['Exited']*100,
              color=['steelblue','tomato','mediumseagreen'])
axes[0,2].set_xlabel('Geography')
axes[0,2].set_ylabel('Churn Rate (%)')
axes[0,2].set_title('Churn Rate by Geography')

# Active Member vs Churn
active_churn = df.groupby('IsActiveMember')['Exited'].mean()
axes[1,0].bar(['Not Active', 'Active'], active_churn.values * 100,
              color=['tomato', 'steelblue'])
axes[1,0].set_ylabel('Churn Rate (%)')
axes[1,0].set_title('Churn Rate: Active vs Not Active')

# Balance Distribution
axes[1,1].hist(df[df['Exited']==0]['Balance'], bins=30, alpha=0.6, label='Retained', color='steelblue')
axes[1,1].hist(df[df['Exited']==1]['Balance'], bins=30, alpha=0.6, label='Churned', color='tomato')
axes[1,1].set_xlabel('Balance')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Balance Distribution by Churn')
axes[1,1].legend()

# Num of Products vs Churn
prod_churn = df.groupby('NumOfProducts')['Exited'].mean()
axes[1,2].bar(prod_churn.index.astype(str), prod_churn.values * 100, color='orchid')
axes[1,2].set_xlabel('Number of Products')
axes[1,2].set_ylabel('Churn Rate (%)')
axes[1,2].set_title('Churn Rate by Num of Products')

plt.tight_layout()
plt.savefig('/home/claude/task3_churn_eda.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 5. Model Training ────────────────────────────────────────
features = ['CreditScore', 'Geography_enc', 'Gender_enc', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
X = df[features]
y = df['Exited']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── 6. Evaluation ────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 55)
print(f"MODEL ACCURACY: {acc*100:.2f}%")
print("=" * 55)
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# Feature Importance + Confusion Matrix plot
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Task 3: Model Evaluation - Churn Prediction", fontsize=13, fontweight='bold')

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[0],
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'])
axes2[0].set_xlabel('Predicted')
axes2[0].set_ylabel('Actual')
axes2[0].set_title(f'Confusion Matrix (Accuracy: {acc*100:.2f}%)')

# Feature Importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
axes2[1].barh(importances.index, importances.values, color='steelblue')
axes2[1].set_xlabel('Importance Score')
axes2[1].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('/home/claude/task3_churn_model.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[Task 3] Visualizations saved!")
print("\nKEY INSIGHTS:")
print("- Older customers (age > 45) churn more frequently")
print("- German customers have the highest churn rate")
print("- Inactive members are significantly more likely to churn")
print("- Customers with 3+ products have higher churn")
