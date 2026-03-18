# Algorithmic Fairness: Mitigating Gender Bias

This repository contains the Project Work developed for the Second-Level Master's in Machine Learning and Big Data at the Università di Padova. The primary objective is to measure and mitigate systematic errors (algorithmic bias) that lead to unfair outcomes, specifically focusing on the disparity of error rates between genders.



### Project Structure

* `code_project_work.py`: The main Python script containing the experimental pipeline:
    * **Data Ingestion**: Loading and parsing the ARFF format dataset.
    * **Exploratory Data Analysis**: Visualizing distributions for sensitive attributes like Sex, Race, and Native Country.
    * **Model Training & Evaluation**: Training models including Gaussian Naive Bayes (GNB), Multi-Layer Perceptron (MLPC), and Support Vector Classifier (SVC).
    * **Bias Measurement**: Plotting feature differences to compare Accuracy, Positive Rate, and Negative Rate disparities between models.
* `Project_work_slides_Gianluca_Ercolani.pdf`: Presentation slides detailing the theoretical framework, dataset selection, and the "Debiasing by unawareness" technique.

### Dataset

The experiments are conducted on the **ACSIncome dataset**, consisting of 48,842 observations. It serves as a modern alternative to the classic UCI Adult dataset for evaluating fairness in machine learning.
Key protected/sensitive attributes analyzed include:
* `SEX`
* `RAC1P` (Race)
* `POBP` (Place of Birth)

### Results & Conclusions
The application of the **Debiasing by unawareness** model-based technique successfully produced a reduction of the gender gap in 4 out of the 5 metrics considered. However, the study also acknowledges that this technique may fall short when other non-protected features highly correlate with the protected attribute.
