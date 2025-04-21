# Nama : Cresenshia Hillary Benida
# NIM : 2702247683
# Dataset A (Loan)

# OOP
# Seluruh proses training dari algoritma machine learning yang terbaik dibubah dalam format OOP


# import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle

# Class DataHandler 
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None  
        self.output_df = None

    # Data read csv
    def load_data(self):
        self.data = pd.read_csv(self.file_path, delimiter=',')

    # Remove feature
    def remove_column(self, column):
        self.data = self.data.drop(columns=[column])
    
    # Data target column
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

    # Handle value yang inkonsisten
    def replace_values(self, column_name, mapping_value):
        if column_name in self.data.columns:
            self.data[column_name] = self.data[column_name].astype(str).str.strip().str.lower()     # seragamkan dulu valuenya
            self.data[column_name] = self.data[column_name].replace(mapping_value)

# Class ModelHandler
class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    # Splitting Data menjadi train dan test data dengan proporsi 80:20
    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
           self.input_data, self.output_data, test_size=test_size, random_state=random_state)
        
    # Check Outliers dengan Boxplot
    def checkTheOutliersWithBox(self, kolom):
        boxplot = self.x_train.boxplot(column=[kolom]) 
        plt.show()

    # Function untuk mengembalikan nilai outliers (digunakan untuk mengisi missing value)
    def check_outliers(self, column):
        return abs(self.x_train[column].skew()) > 1
      
    # Check & Handling missing data
    def checkMissingValues(self, column):
        return self.x_train[column].isnull().sum() > 0

    # Mengisi nilai missing value dengan 3 kondisi dan 3 cara handling yang berbeda
    # Kondisi 1 : missing value di kolom person_income -> isi dengan value pembagian loan_amnt / loan_percent_income
    # Kondisi 2 : missing value di kolom dengan outliers -> isi dengan nilai median train data
    # Konsisi 3 : missing value di kolom tanpa outliers -> isi dengan nilai mean train data
    def fillingNAWithNumbers(self, column):
        # Kondisi 1
        if column == 'person_income':
            # Imput value berdasarkan column loan_amnt dan loan_percent_income
            missing_row_income = self.x_train[column].isna()
            self.x_train.loc[missing_row_income, column] = (
                self.x_train.loc[missing_row_income, 'loan_amnt'] /
                self.x_train.loc[missing_row_income, 'loan_percent_income']
            )
            missing_row_income_2 = self.x_test[column].isna()
            self.x_test.loc[missing_row_income_2, column] = (
                self.x_test.loc[missing_row_income_2, 'loan_amnt'] /
                self.x_test.loc[missing_row_income_2, 'loan_percent_income']
            )
            # handle jika ada nilai pembagi yang 0 sehingga menghasilkan value person_income infinite
            self.x_train[column].replace([np.inf, -np.inf], np.nan, inplace=True)
            self.x_test[column].replace([np.inf, -np.inf], np.nan, inplace=True)
            median_income_train = self.x_train[column].median()
            median_income_test = self.x_test[column].median()
            self.x_train[column].fillna(median_income_train, inplace=True)
            self.x_test[column].fillna(median_income_test, inplace=True)
        else:
            # Kondisi 2
            if self.check_outliers(column):
                fill_value = self.x_train[column].median()
            # Kondisi 3
            else:
                fill_value = self.x_train[column].mean()
            self.x_train[column].fillna(fill_value, inplace=True)
            self.x_test[column].fillna(fill_value, inplace=True)

    # Mengisi missing value feature categorical
    def fillingNAWithCategory(self, column_name):
        mode_value = self.x_train[column_name].mode()[0]
        self.x_train[column_name].fillna(mode_value, inplace=True)
        self.x_test[column_name].fillna(mode_value, inplace=True)

    # Encode feature untuk modeling
    def encode_columns(self,
        binary_columns: dict = None,
        label_columns: dict = None,
        one_hot_columns: list = None):
        # Binary encoding
        if binary_columns:
            self.x_train = self.x_train.replace(binary_columns)
            self.x_test = self.x_test.replace(binary_columns)
            
        # Label encoding
        if label_columns:
            for col, mapping in label_columns.items():
                self.x_train[col] = self.x_train[col].map(mapping)
                self.x_test[col] = self.x_test[col].map(mapping)
                
        # One-hot encoding
        if one_hot_columns:
            for col in one_hot_columns:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encode_train = pd.DataFrame(encoder.fit_transform(self.x_train[[col]]), columns=encoder.get_feature_names_out([col]))
                encode_test = pd.DataFrame(encoder.fit_transform(self.x_test[[col]]), columns=encoder.get_feature_names_out([col]))
                encode_train.index = self.x_train.index
                encode_test.index = self.x_test.index
                self.x_train = pd.concat([self.x_train.drop(columns=[col]), encode_train], axis=1)
                self.x_test = pd.concat([self.x_test.drop(columns=[col]), encode_test], axis=1)
                
        # memastikan semua kolom categorical berubah menjadi numerical
        self.x_train = self.x_train.apply(pd.to_numeric, errors='raise')
        self.x_test = self.x_test.apply(pd.to_numeric, errors='raise')

    # Membuat model dengan XGBoost (model dengan akurasi terbaik)
    def createModel(self):
         self.model = xgb.XGBClassifier()

    # Train model XGBoost
    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    # Membuat prediksi dengan model yang telah dilatih
    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 

    # Membuat report untuk melihat performa model
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict))

    # Menyimpan model dalam bentuk pickle
    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file: 
            pickle.dump(self.model, file) 



file_path = 'Dataset_A_loan.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()

mapping_value_gender = {
    'fe male': 'female',
    'female': 'female'
}
data_handler.replace_values('person_gender', mapping_value_gender)

data_handler.create_input_output('loan_status')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()

for column_name in input_df.columns:
    # menampilkan boxplot untuk column numerical saja
    if input_df[column_name].dtype in ['int64', 'float64']:
        model_handler.checkTheOutliersWithBox(column_name)
    # Check dan isi missing values
    if model_handler.checkMissingValues(column_name):
        # numerical column
        if input_df[column_name].dtype in ['int64', 'float64']:
            model_handler.fillingNAWithNumbers(column_name)  
        # categprical column
        else:
            model_handler.fillingNAWithCategory(column_name) 

model_handler.encode_columns(
    binary_columns={
        'person_gender': {'male': 1, 'female': 0},
        'previous_loan_defaults_on_file': {'Yes': 1, 'No': 0}
    },
    label_columns={
        'person_education': {
            'high school': 0,
            'associate': 1,
            'bachelor': 2,
            'master': 3,
            'doctorate': 4
        }
    },
    one_hot_columns=['person_home_ownership', 'loan_intent']
)

print("XGBoost Model")
model_handler.train_model()

model_handler.makePrediction()
model_handler.createReport()
# model_handler.save_model_to_file('xgb_class.pkl') 

