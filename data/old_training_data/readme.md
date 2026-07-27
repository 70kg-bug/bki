PLEASE RUN THE full.ipynb FILE LOCATED INSIDE /source IF YOU WANT TO TRAIN THE MODEL

All the codes are located inside the /source folder

The /source/visualzed folder contains visualized data of 7 patients' dataset all titled patient 1,...,patient 7 respectively

There are 3 python notebooks inside the /source folder: full.ipynb, GoogleBrainData.ipynb and visualizing_data.ipynb

The full.ipynb notebook contains the code for the whole process. 
1. Starting from import required libraries, turning .xlsx data tables to pandas Dataframes,
2. then merging the Dataframes, processing the data: dropping columns, filtering out uneeded features,... 
3. then I perform splitting the data into training and testing sets.
4. The model chooses for this project is Gradient Boosted Regressor (or Gradient Boosted Trees).
5. After that, I used grid_search to test out every possible hyperparamters combinations to choose the best parameter for the model.
6. Lastly, I ran the metrics to evaluate the model.

The GoogleBrainData.ipynb notebook doesn't serve any functional feature for this project. It contains functions and methods from my past project, I put this notebook in this project's folder for copy paste convinience.

The visualizing_data.ipynb notebook contains codes which serve visualizing purposes. It visualizes 7 pateints' waveform data using matplotlib library.

Inside /patient_information, there are 5 .xlsx files. They are: Clinical Diagnosis and Text Data Table.xlsx, Laboratory Indicators Table.xlsx, Mechanical Ventilation.xlsx, Patient Demographic Information Table.xlsx, Vital Signs and Blood Gas Analysis Table.xlsx.
* Clinical Diagnosis and Text Data Table.xlsx
contains patients' clinical diagnosis and text data. They are all unstructured data which are mostly doctors' notes, hospital records. And they are all written in **Chinese**
* Laboratory Indicators Table.xlsx
contains patients' health statistics
* Mechanical Ventilation.xlsx
contains vetilator's setting for each patients
* Patient Demographic Information Table.xlsx
contains patients' demographic information
* Vital Signs and Blood Gas Analysis Table.xlsx
contains contains patients' health statistics

Inside /waveform_data, there are 7 .xlsx files. They are data files for 7 patients. I mainly used the data from these files to train the model.

