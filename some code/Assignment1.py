## In this assignment, you are going to practice on using the DataFrame in the Pandas package.
## Download the file PastHire.csv from Blackboard. Make sure to memorize the path where you
## downloaded the file.
## Then write the corresponding python instruction that answer or perform each comment.

## Import the pandas as pd
import pandas as pd

## read the file PastHire.csv using pandas read_csv. Do not use index_col=0
## store the data set in a variable called hire.
df= pd.read_csv('data/pastHire.csv', sep= ',', encoding='utf8')

## type hire.info(), this displays the column names and their data type.
print(hire.info())
## type hire.head() to display the first five rows
print(hire.head())
## type hire.head(3) to display the first three rows
print(hire.head(3))

## To select columns from hire, use hire['column name']
## if you have multiple columns, then use: hire[['col1', 'col2', ...]]
print(hire['1'])
print (hire[['col1','col12']])
## Select the column 'Years Experience'
print (hire.loc['years Experience'])

## Select the columns 'Years Experience' and 'Hired'
print (hire.loc[['col1', 'col7']])

## Select the columns 'Years Experience', 'Employed', and 'Hired'
print(hire.loc[['col1', 'col2', 'col7']])
## To display rows, use the hire.loc(row_selection, col_selection)
## to select the rows with index 2, type: hire.loc[2, :] or simply hire.loc[2]
print (hire.loc(row_selection, col_selection))
print (hire.loc[2])

## To display row index 2 and column name 'Employed', type: hire.loc[2, 'Employed?']
print (hire.loc[2,'Emlioyed?'])

## To display multiple rows and multiple columns, then use the list selection:
## to select row index 2 and 3 for all columns; type: hire.loc[[2, 3], :] or simply hire.loc[[2,3]]
print (hire.loc[[2, 3]])

## To select multiple rows and columns, use the list select:
## To select row index 2, 3 and 4 and columns 'Interned' and 'Hired', type:
## hire.loc[[2, 3, 4], ['Interned', 'Hired']]
print (hire.loc[[2, 3 ,4], ['interned', 'Hired']])

## import the package matplotlib.pyplot as plt
import matplotlib.pyplot as plt

## draw a scatter plot between 'Year Experience' as x-axis and 'Hired' as y-axis
import numpy as np
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.show()



## draw a scatter plot between column 'Years Experience' as x-axis and 'Previous employers' as y-axis
import numpy as np
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.show()
