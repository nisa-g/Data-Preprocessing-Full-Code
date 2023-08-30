############################################# OUTLIER TREATMENTS #################################################

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read csv file
df = pd.read_csv(r"Boston.csv")
df.dtypes

# Outliers treatments 
sns.boxplot(df.dis, color='yellow')
sns.boxplot(df.ptratio, color='orange')  
sns.boxplot(df.lstat, color='blue') 
sns.boxplot(df.age, color='blue')                              # No outliers in age column
sns.boxplot(df.tax, color='blue')                              # No outliers in tax column
plt.show(block=False)                

# Detection of outliers (limits for 'dis' based on IQR)

IQR = df['dis'].quantile(0.75) - df['dis'].quantile(0.25)      # IQR = Q3 - Q1

pi          # upper limit = Q3 + IQR*1.5

# Replace the outliers by the maximum and minimum limit
# Method: Rectify
df['df_replaced'] = pd.DataFrame(np.where(df['dis'] > upper_limit, upper_limit, np.where(df['dis'] < lower_limit, lower_limit, df['dis'])))
sns.boxplot(df.df_replaced, color='yellow')

# Winsorization ('ptratio' column) 
# Method: Retain

# Import library
from feature_engine.outliers import Winsorizer 

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                          tail = 'both', 
                          fold = 1.5, 
                          variables = ['ptratio'])

df_retained = winsor_iqr.fit_transform(df[['ptratio']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Visualization
sns.boxplot(df_retained.ptratio, color='orange')
plt.show(block=False)  

# Define the model with Gaussian method ('lstat' column)
winsor_gaussian = Winsorizer(capping_method = 'gaussian', 
                          tail = 'both', 
                          fold = 2.5,
                          variables = ['lstat'])

df_retained = winsor_gaussian.fit_transform(df[['lstat']])
sns.boxplot(df_retained.lstat, color='blue')
plt.show() 

########################################## DISCRETIZATION ##########################################################

# Import library
import pandas as pd

# Read csv file
data = pd.read_csv(r"iris.csv")
data.dtypes

# Discretization of Sepal_Length, Sepal_Width, Petal_Length and Petal_Width
data['Sepal_Length'] = pd.cut(data['Sepal_Length'], 
                              bins = [min(data.Sepal_Length), data.Sepal_Length.mean(), max(data.Sepal_Length)], 
                              include_lowest = True,
                              labels = ["Low", "High"])

data.Sepal_Length.value_counts()

data['Sepal_Width'] = pd.cut(data['Sepal_Width'], 
                              bins = [min(data.Sepal_Width), data.Sepal_Width.mean(), max(data.Sepal_Width)], 
                              include_lowest = True,
                              labels = ["Low", "High"])

data.Sepal_Width.value_counts()

data['Petal_Length'] = pd.cut(data['Petal_Length'], 
                              bins = [min(data.Petal_Length), data.Petal_Length.mean(), max(data.Petal_Length)], 
                              include_lowest = True,
                              labels = ["Low", "High"])

data.Petal_Length.value_counts()

data['Petal_Width'] = pd.cut(data['Petal_Width'], 
                              bins = [min(data.Petal_Width), data.Petal_Width.mean(), max(data.Petal_Width)], 
                              include_lowest = True,
                              labels = ["Low", "High"])

data.Petal_Width.value_counts()

# Exporting discretized dataset
output_file = 'discretized_iris.csv'
data.to_csv(output_file, index=False)
print(f"DataFrame exported to {output_file}")

output_file = "C:\Data Sets\discretized_iris.csv"

#################################################### DUMMY VARIABLES ######################################################

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read datasets
df = pd.read_csv(r"Animal_category.csv")

df.columns 
df.shape
df.dtypes
df.info()

# Create dummy variables 
df_new = pd.get_dummies(df)

df_new_1 = pd.get_dummies(df, drop_first = True)

# One Hot Encoding 
df.columns
df = df.drop(['Index'], axis = 1)

# Creating instance of One-Hot Encoder, convert data to numerical format
from sklearn.preprocessing import OneHotEncoder

# initializing method, array font, matrix format
enc = OneHotEncoder() 

enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 2:]).toarray())   # access only certain column iloc(row, column) : means take all rows, for column take from 2 to all 

################################################### DUPLICATION & TYPECASTING ##########################################

# Q1: Typecasting

# Read csv file
import pandas as pd
data = pd.read_csv(r"OnlineRetail.csv", encoding='unicode_escape')
data.dtypes

'''
CustomerID is an integer - Python automatically identify the data types by interpreting the values. 
As the data, these columns are numeric, Python detects the values as float64 in this case.

From measurement levels prespective, these ID is a Nominal data as they are an identity for each entries.
We have to alter the data type which is defined by Python as astype() function

'''

# Convert 'float64' to 'str' (string) type. 
data.CustomerID = data.CustomerID.astype('str')
data.dtypes

# Q2: Duplicates

# Duplicates in rows
duplicate = data.duplicated()                           # Boolean series to identify duplicate rows.
duplicate

sum(duplicate)                                          # Total duplicates in dataset  

# Removing Duplicates
data1 = data.drop_duplicates() 
duplicate = data1.duplicated()
sum(duplicate)                                          # Total duplicates = 0

# Q3: Data analysis (EDA)

# Impute missing values
data1.isna().sum()
data1["Description"] = pd.DataFrame(random_imputer.fit_transform(data1[["Description"]]))
data1["Description"].isna().sum()  

# Dataprep

from dataprep.eda import create_report
report = create_report(data1, title = 'Online Retail Analysis')
report.show_browser()
  
################################################## MISSING VALUES #########################################################

# Import libraries
import numpy as np
import pandas as pd

# Load claimants dataset
df = pd.read_csv(r"claimants.csv") 

# Check for count of NA's in each column
df.isna().sum()

# Imputer object will be created to fill 'Nan' values
# Mean and Median imputer are used for numeric data (CLMAGE, CLMINSUR, SEATBELT)
# Mode is used for discrete data (CLMSEX)

# For Mean, Median, Mode imputation we use Simple Imputer 
from sklearn.impute import SimpleImputer

# Mean Imputer,  The mean imputation can be sensitive to outliers since it pulls the imputed values towards the mean.
mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') 

df["CLMINSUR"] = pd.DataFrame(mean_imputer.fit_transform(df[["CLMINSUR"]]))
df["CLMINSUR"].isna().sum()   

df["SEATBELT"] = pd.DataFrame(mean_imputer.fit_transform(df[["SEATBELT"]]))  
df["SEATBELT"].isna().sum()                                                           # all missing records replaced by mean 

# Median Imputer, it can be less sensitive to outliers since the median is a robust measure of central tendency.
median_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')

df["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(df[["CLMAGE"]]))
df["CLMAGE"].isna().sum()                                                             # all missing records replaced by median 

# Mode Imputer,This technique is used for categorical data or discrete variables. 
mode_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

df["CLMSEX"] = pd.DataFrame(mode_imputer.fit_transform(df[["CLMSEX"]]))
df["CLMSEX"].isna().sum()                                                             # all missing records replaced by mode

# Random Imputer
# automaticallly take median value
from feature_engine.imputation import RandomSampleImputer

random_imputer = RandomSampleImputer(['CLMAGE'])
df["CLMAGE"] = pd.DataFrame(random_imputer.fit_transform(df[["CLMAGE"]]))
df["CLMAGE"].isna().sum()                                                             # all records replaced by median

################################################### STRING MANIPULATION ###########################################################

# Q1

# Create a string "Grow Gratitude"
string = "Grow Gratitude"

# a) Access the letter "G" of "Growth"
first_letter = string[0]
print("The first letter of 'Grow Gratitude' is:", first_letter)

# b) Length of the string
length = len(string)
print("The length of the string 'Grow Gratitude' is:", length)

# c) Count of "G" in the string
count = string.count('G')
print("The number of occurrences of 'G' in 'Grow Gratitude' is:", count)

# Q2

# a) Count number of characters in the string.

s = "Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else."

print(s.count(' '))

# Q3 

# Create a string
s = "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"

# a) one char of the word
print(s[0:1])

# b) first three char
print(s[0:3])

# c) last three char
print(s[-3:])

# Q4

# Create a string
s = "stay positive and optimistic"

# a) string starts with “H”
print(s.startswith("H"))

# b) string ends with “d”
print(s.endswith("d"))

# c) string ends with “c”
print(s.endswith("c"))

# Q5

# code to print " 🪐 " 108 times

planet = "🪐"

print(planet*108)

# Q6

# Create a string
s = "Grow Gratitude"

# Replace “Grow” with “Growth of”
s.replace("Grow", "Growth of")

# Q7

# Create a string
s = ".elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A"

# Story in the correct order
print("".join(reversed(s)))

################################################# STANDARDIZATION & NORMALIZATION ##################################################

# Import libraries
import pandas as pd
import numpy as np

# Read the numerical dataset
data = pd.read_csv(r"Seeds_data.csv")
a = data.describe()

# Standardization
from sklearn.preprocessing import StandardScaler

# Initialise the Scaler
scaler = StandardScaler()

# To scale data
df = scaler.fit_transform(data)                                  # mean = 0

# Convert the array back to a dataframe
dataset = pd.DataFrame(df)
res = dataset.describe()                                         # mean 0 or less than 0, std = 1

# Normalization
from sklearn.preprocessing import MinMaxScaler
minmaxscale = MinMaxScaler()

# To normalize the data
df_n = minmaxscale.fit_transform(df)
dataset1 = pd.DataFrame(df_n)

res1 = dataset1.describe()                                      # min = 0 max = 1

################################################### TRANSFORMATIONS #############################################################

# Normal Quantile-Quantile Plot

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import numpy as np
import seaborn as sns

# Read dataset
cal = pd.read_csv(r"calories_consumed.csv")

# Checking whether data is normally distributed
stats.probplot(cal.Weight_gained, dist = "norm", plot = pylab)     # if data is falling in the straight line so it is normal dist
plt.show()
stats.probplot(cal.Calories_consumed, dist = "norm", plot = pylab) # not normally dist
plt.show()

# Transformation to make weight_gained variable normal
stats.probplot(np.log(cal.Weight_gained), dist = "norm", plot = pylab)
plt.show()

# Boxcox method to visualize the transformation

# Original data
prob = stats.probplot(cal.Weight_gained, dist = stats.norm, plot = pylab)

# Transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(cal.Weight_gained)

# creating axes to draw plots
fig, ax = plt.subplots(1, 2)

# Plotting the original data (non-normal) and fitted data (normal)
sns.distplot(cal.Weight_gained, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Non-Normal", color = "blue", ax = ax[0])

sns.distplot(fitted_data, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Normal", color = "blue", ax = ax[1])

# adding legends to the subplots
plt.legend(loc = "upper right")

# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)

print(f"Lambda value used for Transformation: {fitted_lambda}")

# Transformed data
prob = stats.probplot(fitted_data, dist = stats.norm, plot = pylab)
plt.show()

################################################### ZERO VARIANCE ###########################################################

# Import library
import pandas as pd

# Read dataset
df = pd.read_csv(r"Z_dataset.csv")
df.dtypes
 
# Checking for variation in the data
df.var() 
df.var() == 0
df.var(axis = 0) == 0

# There is no zero variance in the columns dataset.
# In other case, if the variance is low or close to zero, then a feature is approximately constant and will not improve the performance of the model.
# In that case, it should be removed.
