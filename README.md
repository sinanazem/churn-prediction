# Churn Prediction
<img src="https://blog.usetada.com/hubfs/Customer%20Churn%20and%20How%20to%20Stop%20It.png#keepProtocol">
This project aims to provide a solution for predicting customer churn in a telecommunications company by integrating a range of technologies. First, we will develop a machine learning model using Catboost by studying churn data. We will then use Streamlit to present the outputs of this model through a user-friendly interface. Following this, we will build an API using FastAPI to provide real-time predictions, and finally automate the deployment and scaling of the entire solution using Docker.  

### Project Steps:  
1- Data Preprocessing and Model Deployment (CatBoost)  
2- Interface (Streamlit)  
3- API (FastAPI)  
4- Automation (Docker)  
1. Data Preprocessing and Model Development (Catboost)  
- We will perform data preprocessing steps to prepare the dataset with customer information such as account and demographics for model training  
- For a data set where we have a lot of categorical variables, it would be a good choice to use the CatBoost model, which has shown success in categorical variables  
2. Interface (Streamlit)  
- Develop an interactive web interface to visualize and share model outputs  
- Also a small predict facility in this interface  
3. API (FastAPI)  
- Creating an API using FastAPI to enable real-time predictions of the model  
- Providing APls for integration with other systems and automated decision-making processes  
4. Automation (Docker)  
- Packaging the entire solution into a container using Docker  
- Leveraging Docker to automate deployment processes and enable easy deployment of the solution to different environments
