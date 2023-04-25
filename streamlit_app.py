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
    
    # Check for feature types
    feature1_type = df[feature1_dropdown].dtype
    feature2_type = df[feature2_dropdown].dtype
    
    # Determine appropriate correlation metric
    if feature1_type == 'object' or feature2_type == 'object':
        st.write("Selected features are categorical. Pearson correlation is not a good choice because it is typically used for continuous features to measure their strength of relationship.")
        
    else:
        if metric_dropdown == 'pearson':
            st.write("Both of the features are continuous. Therefore, Pearson correlation would give much better results because Pearson correlation coefficient is used to measure the strength and direction of a linear relationship between two continuous variables. It assumes that the variables have a normal distribution.")
        elif metric_dropdown == 'kendall':
            st.write("Selected features are continuous. Kendall correlation is appropriate because Kendall correlation coefficient is a non-parametric measure of rank correlation. It is used to measure the strength and direction of a monotonic relationship between two variables and it is appropriate for both continuous and ordinal variables.")
        else:
            st.write("Selected features are continuous. Spearman correlation is appropriate because Spearman correlation coefficient is a non-parametric measure of rank correlation, which means it is used to measure the strength and direction of a monotonic relationship between two variables. It is appropriate for both continuous and ordinal variables. Though, we can use Pearson metric for better results because it better suits continuous features.")
    
    # Compute the correlation matrix
    corr = df[[feature1_dropdown, feature2_dropdown]].corr(method=metric_dropdown)
    
    # Display the heatmap using seaborn
    st.write("Correlation Matrix")
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot()
    
    
if __name__ == '__main__':
    main()
