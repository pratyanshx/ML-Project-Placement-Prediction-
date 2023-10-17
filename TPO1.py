# FINAL CODE FOR ML PROJECT (PLACEMENT)

# PLACEMENT PREDICTION MODEL
# STUDENTS PLACED V/s YEAR

# Importing libraries
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import tkinter as tk

# Loading the dataset
data = pd.read_csv("D:/Coding/CSV Files/PYY.csv", sep=",")

# EDA - Exploratory Data Analysis
data = data[['No of Students Placed','Years']]

# Changing format (array) (ML PART)
x = np.array(data['Years']).reshape(-1,1)
y = np.array(data['No of Students Placed']).reshape(-1,1)

# Plotting the graph 
plt.xlabel("Years")
plt.ylabel("No of Students Placed")
plt.plot(x,y)
plt.show()

# Ploynomial Regression
Poly = PolynomialFeatures(degree = 18)
x1 = Poly.fit_transform(y)

# Training the data and preparing the model
model = linear_model.LinearRegression()
model.fit(x1,y)
accuracy = model.score(x1,y)

# print(f'Accuracy:{(round(accuracy*100,2))}%')

# Plotting the test-model data0
y0 = model.predict(x1)
plt.xlabel("Year")
plt.ylabel("No of Students Placed")
plt.plot(x,y0, '--b')                 
plt.show()

# Comparing model data with actual data
plt.plot(x,y0, '--b')
plt.plot(x,y, '-m')
plt.show()

# For taking user input of years in terminal
# years = int(input("Enter the year after 2022 : "))

years = 1

# This is Time Series Data

# For printing values in terminal
# print(f'Prediction - Students placed after {years} years = ', end = ' ')
# print(round(int(model.predict(Poly.fit_transform([[214 + years]])))), 'Students')

x1 = np.array(list(range(1, 214 + years))).reshape(-1, 1)
y1 = model.predict(Poly.fit_transform(x1))

def predict_placements():
    try:
        year = int(year_entry.get())  # Get the year entered by the user
        year_value = np.array([[214 + year]])
        year_value_poly = Poly.transform(year_value)
        predicted_placements = model.predict(year_value_poly)
        result_label.config(text=f"Predicted placements after {year} year: {round(predicted_placements[0][0])}")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter a valid year.")

# Create a function to display accuracy in the GUI window
def display_accuracy():
    accuracy_label.config(text=f"Accuracy: {round(accuracy*100, 2)}%")

# Create a function to exit the GUI
def exit_gui():
    root.destroy()

root = tk.Tk()
root.title("Placement Prediction")

# Create the label and entry for year input
year_label = tk.Label(root, text="Enter the desired year (Year after 2022 in form of 1,2,3...) :")
year_label.pack(padx=10, pady=5)

year_entry = tk.Entry(root, width=10)
year_entry.pack(padx=10, pady=5)

# Create the predict button
predict_button = tk.Button(root, text="Predict", command=predict_placements)
predict_button.pack(padx=10, pady=5)

# Create the result label
result_label = tk.Label(root, text="")
result_label.pack(padx=10, pady=5)

# Create the Display Accuracy button
accuracy_button = tk.Button(root, text="Display Accuracy", command=display_accuracy)
accuracy_button.pack(padx=10, pady=5)

# Create the accuracy label
accuracy_label = tk.Label(root, text="")
accuracy_label.pack(padx=10, pady=5)

# Create the Exit button
exit_button = tk.Button(root, text="Exit", command=exit_gui)
exit_button.pack(padx=10, pady=5)

# Start the GUI event loop
root.mainloop() 