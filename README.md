# Note that the data2 will not be provided directly in this project, you will need to follow the instructions "HOWTO2" down below.




Datasets:

[data.csv] 
name:
link:


[data2.csv]
name: hospital-triage-and-patient-history-data
link: https://www.kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data

Either you download it directly by clicking on the link above, or you can use the following linux command wget (if you are opting for the second method be sure to have the credential json file in ~/.kaggle path)

The data provided is in .rdata format, to convert it into a csv format, the converter code is provided in the tools folder.



Requirements when using the HPC Clusters from Adapt Center for the first time:
    1. Install a recent version of Python
    2. Install a recent version of the C compiler gnn
    3. Install cuda/cuDNN/cuda Toolkit
    



# HOWTO1: Run Llama3 on HPC Adapt Center clusters
    1. It is needed to create a temporary folder as the default tmp folder requires sudo permissions to write in it.




# HOWTO2: Download the data2.RData and converting the data2.RData into a csv file
    1. Go to the link and download the data2 from the link provided above. 
    2. Go to the R file named "ConvertIntoCsv.R" and run the code.

