import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing import load_data, process_features_targets, split_and_scale_data
from src.dimensionality import apply_pca
from src.models_registry import get_model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



st.set_page_config(page_title="ML Models Hub", layout="wide")
st.title("ML Models Hub")

st.sidebar.header("settings")
uploaded_file = st.sidebar.file_uploader("1️⃣ Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader("📋 Initial Look at the Data")
    st.dataframe(df.head())
    
    # select target column from the dataset
    all_columns = df.columns.tolist()
    target_column = st.sidebar.selectbox("Select Target Column:", all_columns, index=len(all_columns)-1)
    
    # process features and targets
    X, y, error_msg = process_features_targets(df, target_column)
    
    if error_msg:
        st.error(error_msg)
    else:
        test_size = st.sidebar.slider("3️⃣ Test Size:", 0.1, 0.5, 0.2, step=0.05)
        X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X, y, test_size)
        
        # تطبيق PCA
        max_components = min(X_train_scaled.shape[1], X_train_scaled.shape[0])
        n_components = st.sidebar.slider("4️⃣ Number of PCA Components:", 1, max_components, min(2, max_components))
        pca, X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled, n_components)
        
        # select model from the registry
        models_dict = get_model()
        selected_model_name = st.sidebar.selectbox("5️⃣ Select Model:", list(models_dict.keys()))
        
        # Train and evaluate the selected model
        model = models_dict[selected_model_name]
        model.fit(X_train_pca, y_train)
        
        y_pred = model.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display results
        st.subheader(f"🎯 Results of Using Model: {selected_model_name}")
        st.metric(label="Overall Model Accuracy", value=f"{accuracy*100:.2f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("📊 Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=np.unique(y), yticklabels=np.unique(y))
            ax_cm.set_xlabel('Predicted labels')
            ax_cm.set_ylabel('True labels')
            st.pyplot(fig_cm)
            
        with col2:
            st.write("📝 Detailed Performance Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='bone'))
else:
    st.info("💡 Please upload a data file and start experimenting with different models from the sidebar.")