# Hospital Cost Modeling

This project is the first of several personal data science initiatives I’m adding to my portfolio outside of academic work. It focuses on understanding and predicting hospital costs using real-world healthcare data.

---

## **📊 About the Dataset**
The [dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) used in this project contains medical insurance cost data for individuals. It includes the following features:

- **age:** Age of the individual.
- **sex:** Gender of the individual.
- **bmi:** Body Mass Index (BMI), a measure of body fat based on height and weight.
- **children:** Number of children/dependents covered by the insurance plan.
- **smoker:** Whether the individual is a smoker or not.
- **region:** The individual's residential region in the United States.
- **charges:** The medical insurance cost charged to the individual (target variable).
This dataset allows us to explore how various factors like age, BMI, smoking habits, and region contribute to medical insurance costs. It is particularly useful for regression tasks, as we aim to predict the charges based on the other features.

---

## 📄 Project Overview

The analysis is structured in **four phases** across 3 documents:
Render PDF and HTML of both markdowns can be found in `docs` folder.
- dataExploration.qmd
- dataModeling.qmd
- app.py


### 1. Data Exploration
- Examine the dataset to understand **univariate relationships** (individual features).  
- Investigate **multivariate relationships** to uncover correlations and patterns between variables.  
- Identify key trends and insights to inform predictive modeling.

### 2. Data Visualization
Visual exploration of univariate and bivariate relationships to detect patterns, anomalies, and potential modeling challenges.

#### 2.1 Univariate Analysis
Analysis of individual variables to understand their distributions and characteristics.

**Numerical Features**
- Distribution shape (histograms, KDEs)  
- Central tendency and spread  
- Outlier detection  
- Target variable (`charges`) distribution

**Categorical Features**
- Frequency distributions  
- Class imbalance inspection  
- Discrete structures (e.g., number of children, regions)

#### 2.2 Bivariate Analysis
Examine relationships between predictors and the target variable (`charges`).

**Categorical vs Target**
- Sex vs charges  
- Smoker status vs charges  
- Region vs charges  
- Family size vs charges

**Numerical vs Target**
- Age vs charges  
- BMI vs charges  
- Subgroup effects (e.g., smoker vs non-smoker)

### 3. Predictive Modeling
Build and evaluate **two predictive models** for hospital cost estimation:  
- Multiple Linear Regression  
- Random Forest

#### 3.1 Hyperparameter Tuning
- Models were evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² on both training and test sets.  
- The Optuna library was used to tune hyperparameters to minimize the loss function.

#### 3.2 Feature Importance
**SHAP Feature Importance:**  
- SHAP values were used to interpret the final model’s predictions and rank feature importance.  
- This helps identify the key drivers behind medical cost predictions and potential applications in healthcare analytics.

#### 3.3 Determine Final Model
- Compare model performance using appropriate metrics to select the best candidate for deployment.  
- **Save Final Model:** The optimized model was saved for future use.

### 4. Model Deployment
- Real-time predictions using the tuned Random Forest pipeline (preprocessing + model)
- Prediction-specific 95% confidence intervals calculated from individual tree predictions
- Visual representation of prediction ranges with risk factor insights
- Responsive UI with input validation for all demographic and health features

*["img/modeling_screenshot.png"]*


---

## Tools & Technologies
- Python (`pandas`, `numpy`, `scikit-learn`, `optuna`, `matplotlib`, `seaborn`, and `streamlit`)  
- Quarto Notebook for interactive analysis and visualization  
- GitHub for version control and project documentation

---

## 🔍Key Takeaways
- Explored the dataset to gain actionable insights into hospital cost drivers.  
- Developed and evaluated predictive models for real-world healthcare applications.  
- Built a foundation for deploying data-driven solutions in medical cost management.

---

## **📌 Summary**
This project focuses on predicting medical costs through the development and comparison of machine learning models, with a strong focus on optimization and interpretability. Two models—Random Forest and Linear Regression—were trained and evaluated, with Random Forest demonstrating superior predictive performance. Hyperparameter tuning using Optuna further enhanced model accuracy, while SHAP analysis provided a detailed understanding of feature contributions. Key factors such as smoking status, BMI, and age were identified as major drivers of medical costs. The workflow encompassed the entire process, from exploratory data analysis and feature evaluation to model training, optimization, and final selection, culminating in a saved, deployable model.

---

To deploy the UI, the following bash script can be used with `uv` package manager. If `uv` is not installed, instructions can be found [here](https://docs.astral.sh/uv/getting-started/installation/).
```bash
git clone https://github.com/JustSplash8501/hospital-cost-modeling.git
cd hospital-cost-modeling

uv init
uv sync
uv run python
uv run streamlit run app.py
```
The app will automatically open in your browser at `http://localhost:8501`