# CapstoneFlightPrices2
Second and final part of AI/ML Professional Certificate Capstone, doing ML operations on a large set of Indian flights data to predict prices against duration/days until departure


## Flight Prices Databricks Machine Learning

## Project Overview

This project aims to leverage the powerful analytics and machine learning capabilities of Databricks to predict flight prices based on historical data. Utilizing Apache Spark for large-scale data processing and PySpark for machine learning, the project provides insights into factors affecting flight prices and predicts future trends. The code repository can be found here: https://github.com/AnnikaPeacockAI/CapstoneFlightPrices1/blob/main/FlightPrices_Databricks_MachineLearning.ipynb

## Objective
The primary goal is to develop a predictive model that can accurately forecast the prices of flights based on various input variables such as departure date, airline, flight duration, number of stops, and other relevant factors. This model can help consumers and businesses make informed decisions by understanding price dynamics in the aviation industry.

## Methodology
The project follows these steps:

Data Acquisition: Collect historical flight price data from various sources. This dataset includes features like flight dates, airlines, destinations, ticket prices, and more.

Data Preparation and Exploration: Cleanse and preprocess the data, followed by exploratory data analysis to understand patterns and correlations.

Feature Engineering: Create new features that can improve the model's predictive power based on the initial analysis.

Model Building: Utilize PySpark's machine learning libraries to train multiple regression models, comparing their performance to select the best one.

Evaluation and Tuning: Assess the model's accuracy and fine-tune its parameters using cross-validation and other optimization techniques.

Deployment: Implement the model within a Databricks notebook to provide real-time or batch predictions on flight prices.

Visualization: Create dashboards and visualizations to present the results and findings, aiding in the interpretability of the predictive models.

## Tools and Technologies Used
Databricks: For running Spark clusters and executing notebooks that contain the project's code.
Apache Spark & PySpark: For data processing and machine learning tasks.
Matplotlib, Seaborn, Pandas: For data visualization and manipulation in Python.

## Machine Learning Model Training and Evaluation
In this project, we have employed three different regression models to predict flight prices: Linear Regression, Random Forest Regressor, and Gradient Boosted Trees (GBT) Regressor. The performance of each model was evaluated based on the Root Mean Square Error (RMSE) metric on test data.

Linear Regression Model
RMSE on Test Data: 2942.37
Best Parameters:
Aggregation Depth: 2
Elastic Net Param: 0.0
Epsilon: 1.35
Fit Intercept: True
Loss: squaredError
Max Iterations: 100
Regularization Param: 0.0
Tolerance: 1e-06
This model serves as a baseline for our predictive task.

Random Forest Regressor
RMSE on Test Data: 2490.67
Best Parameters:
Bootstrap: True
Max Depth: 5
Max Bins: 32
Num Trees: 20
Feature Subset Strategy: auto
The Random Forest model provided a significant improvement over the baseline Linear Regression model.

Gradient Boosted Trees (GBT) Regressor
RMSE on Test Data: 2467.86
Best Parameters:
Loss Type: squared
Max Depth: 5
Max Bins: 32
Max Iterations: 20
Step Size: 0.1
The GBT Regressor yielded the best performance among the three models, making it our preferred choice for predicting flight prices.

## Advanced Analytics and Utility Scores
Beyond basic price predictions, this project also incorporates utility scores and user-defined preferences for flight duration and cost. This allows for personalized recommendations that balance between price and flight duration, catering to different traveler profiles. This aspect of the project utilizes user inputs to weigh various factors more significantly according to individual preferences. This feature will be expanded upon in the next phase of the project, aiming to deliver more tailored and practical flight recommendations.



## Getting Started

This project uses Databricks and machine learning techniques to predict flight prices. We perform data preprocessing, exploratory data analysis (EDA), and model training using historical flight data to forecast prices effectively.

### Prerequisites

Before you begin, ensure you meet the following requirements:

Databricks Environment: Access to Databricks with permissions to create and run clusters. Ensure your Databricks setup includes Apache Spark. While this project is compatible with most modern Spark versions, it is tested with Spark 3.x.

Apache Spark & PySpark: The project is developed using Apache Spark. Ensure that your Databricks cluster is running a compatible version of Spark (preferably 3.x). PySpark is used for data processing and machine learning operations. It should be automatically available in your Databricks environment.

Python: The project uses Python for data visualization and additional data manipulation. Ensure Python 3.6 or newer is installed in your environment. The project is tested on Python 3.8.

Python Libraries: The following Python libraries are used:

matplotlib and seaborn for data visualization.
pandas for data manipulation and analysis.
These libraries can be installed using pip:

```
pip install matplotlib seaborn pandas
```

In Databricks, you can install these libraries directly to your cluster or use a notebook to install them for the session.

Machine Learning Libraries: The project utilizes the machine learning libraries provided by PySpark. No additional installation is required as these are part of the PySpark distribution within Databricks.

Ensure that all the prerequisites are met before proceeding with the installation and execution of the project. If you encounter any issues with the prerequisites, refer to the respective documentation of each component or contact your Databricks administrator.

### Installing

AFollow these steps to set up your development environment:

Step 1: Log into Databricks
Log into your Databricks workspace. If you don't have an account, you will need to create one.

Step 2: Create a New Cluster in Databricks:

```
1. Navigate to the 'Clusters' section from the sidebar.
2. Click 'Create Cluster'.
3. Name your cluster.
4. Select a Databricks Runtime Version that includes Apache Spark (we recommend the latest version supporting Spark 3.x).
5. Click 'Create Cluster' and wait for it to start.
```

Step 3: Creating a New Notebook for your Project
```
1. Navigate to the 'Workspace' section from the sidebar.
2. Choose or create a new folder.
3. Click 'Create' > 'Notebook'.
4. Name your notebook and set 'Python' as the default language.
5. Assign the notebook to your newly created cluster.
```

Step 4: Import Project Code
Copy the project code into the notebook or import the notebook directly if available.

Step 5: Uploading the Dataset

This project is based on the Flight Price Prediction dataset found on Kaggle: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

Once you have downloaded the dataset, you need to upload it to Databricks:

-Navigate to the 'Data' section in your Databricks workspace.
-Choose 'Add Data', and then upload the dataset file from your local machine.
-Once uploaded, note the file path provided by Databricks for the dataset.

```
# Replace 'filePath' with the path provided by Databricks after uploading the dataset
dataset_path = "dbfs:/FileStore/tables/your_dataset.csv"  # Update this with your actual file path

# Load the dataset
df = spark.read.csv(dataset_path, header=True, inferSchema=True)
df.show()
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details
