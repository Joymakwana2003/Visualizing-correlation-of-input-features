import pandas as pd
import seaborn as sns
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
# Load the dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
df.columns = ["ID", "Diagnosis", "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
              "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
              "SE Radius", "SE Texture", "SE Perimeter", "SE Area", "SE Smoothness", "SE Compactness", "SE Concavity",
              "SE Concave Points", "SE Symmetry", "SE Fractal Dimension", "Worst Radius", "Worst Texture",
              "Worst Perimeter", "Worst Area", "Worst Smoothness", "Worst Compactness", "Worst Concavity",
              "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"]

# Define the main function
def main():
    # Set the title and the page layout
    st.set_page_config(page_title="Breast Cancer Wisconsin Diagnostic Dataset")
    st.title("Breast Cancer Wisconsin Diagnostic Dataset")
    
    # Define the widgets
    feature1_dropdown = st.selectbox('Select feature 1:', options=list(df.columns[2:]))
    feature2_dropdown = st.selectbox('Select feature 2:', options=list(df.columns[2:]))
    metric_dropdown = st.selectbox('Select metric:', options=['pearson', 'kendall', 'spearman'])
    
    # # Check for duplicate column names
    # if feature1_dropdown == feature2_dropdown:
    #     st.warning("Please select a different feature for Feature 2.")
    #     return
    
    # Compute the correlation matrix
    corr = df[[feature1_dropdown, feature2_dropdown]].corr(method=metric_dropdown)
    
    # Display the heatmap using seaborn
    st.write("Correlation Matrix")
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot()
    
    
if __name__ == '__main__':
    main()
