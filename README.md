# Email_Fusion

### Overview  

**Email Fusion: Email Filtering System** is a machine learning-based email classification system designed to efficiently filter spam and non-spam emails. It leverages advanced preprocessing, feature extraction, and multiple classification algorithms to ensure accurate filtering.  

The project includes data extraction from Gmail, exploratory data analysis (EDA), and machine learning model training to optimize spam detection.  

### Features  

#### Data Extraction and Processing  
- Extracts emails from Gmail and processes them into structured datasets.  
- Performs text preprocessing using **NLTK** and **Scikit-learn** for feature extraction.  

#### Machine Learning Models  
Trained and evaluated multiple classifiers to determine the best-performing model:  
- **Random Forest**  
- **Multinomial Naïve Bayes**  
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree Classifier**  
- **Support Vector Machine (SVM - Linear & RBF)**  

The **SVM (RBF) model** achieved the highest accuracy of **97.45%**, making it the optimal choice for spam classification.  

#### File Details  
- **All_Emails.xlsx** – Contains all extracted emails.  
- **non_spam.xlsx** – Dataset with non-spam emails.  
- **spam.xlsx** – Dataset with spam emails.  
- **Email_Fusion.ipynb** – Core implementation of the filtering system.  
- **Extraction of Emails from Gmail.ipynb** – Script for extracting emails from Gmail.  
- **Email_Filtering_System.pdf** – Documentation of the project.  

### Insights Generation  
- Identified key linguistic and statistical patterns in spam emails.  
- Evaluated classification models based on **training performance and processing time**.  

### How to Use  
1. Run `Email_Fusion.ipynb` to preprocess and classify emails.  
2. Use the provided datasets to test filtering accuracy.  
3. Modify or retrain models for further optimization.  

### Dependencies  
- **Python**  
- **Pandas, NumPy, Matplotlib, Seaborn**  
- **NLTK** for text processing  
- **Scikit-learn** for machine learning models  

### Contributing  
Contributions are welcome! Feel free to open issues or pull requests to enhance classification accuracy, optimize processing speed, or add new filtering techniques. Let's collaborate to build a smarter email filtering system!  
