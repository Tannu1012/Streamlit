import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("mail_data (1).csv")

# Clean the data
mail_data = df.where((pd.notnull(df)), '')

# Encode labels
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Features and Labels
X = mail_data['Message']
Y = mail_data['Category'].astype('int')

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Accuracy
train_acc = accuracy_score(Y_train, model.predict(X_train_features))
test_acc = accuracy_score(Y_test, model.predict(X_test_features))

# ---------------- Streamlit App ----------------
st.title("📧 Spam Mail Prediction App")
# st.tiitle("")

st.write(f"Model trained with **{train_acc:.2f}** accuracy on training data and **{test_acc:.2f}** on test data.")
# User input
user_input = st.text_area("Enter the mail/message text:")

if st.button("Predict"):
    if user_input.strip() != "":
        input_features = feature_extraction.transform([user_input])
        prediction = model.predict(input_features)[0]
        if prediction == 0:
            st.error("This message is **SPAM**!")
        else:
            st.success("This message is **HAM (Not Spam)**.")
    else:
        st.warning("Please enter a message to classify.")

st.sidebar.title("ℹ️ About This App")
st.sidebar.write("""
This is a **Spam Mail Prediction App** built with Streamlit.  
It uses **TF-IDF vectorization** and **Logistic Regression** to classify emails/messages as:
- **SPAM** 🚨  
- **HAM (Not Spam)** ✅  

**How to use:**  
1. Enter your message in the text box.  
2. Click **Predict** to see the result.  

**Dataset:** SMS Spam Collection dataset (spam vs ham messages).
""")