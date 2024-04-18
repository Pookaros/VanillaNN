def MSEstep(input:float, target:float, MSE:float, stepno:int):
    ###CASE 0###
    if stepno == 0:
        difference = target - input 
        MSE = difference**2
    ###MAIN CASES###
    else:
        difference = target - input # finding the difference between observed and predicted value
        squared_difference = difference**2  # taking square of the differene
        MSE = MSE*(stepno)/(stepno + 1) + squared_difference/(stepno + 1)  # taking a sum of all the differences
    return MSE

def MAEstep(input:float, target:float, MAE:float, stepno:int):
    ###CASE 0###
    if stepno == 0:
        difference = target - input 
        MAE = abs(difference)
    ###MAIN CASES###
    difference = target - input # finding the difference between observed and predicted value
    squared_difference = abs(difference)  # taking square of the differene
    MAE = MAE*(stepno)/(stepno + 1) + squared_difference/(stepno + 1)  # taking a sum of all the differences
    return MAE

###We will only use the step function in the perceptron implementation after all###

def MSE(inputs, target, indrange:list):
    summation = 0  # variable to store the summation of differences
    n = len(input)  # finding total number of items in list
    for i in range(indrange[0], indrange[1]):  # looping through a selected range from the list
        difference = target[i] - inputs[i] # finding the difference between observed and predicted value
        squared_difference = difference**2  # taking square of the differene
        summation += squared_difference # taking a sum of all the differences
    MSE = summation/n  # dividing summation by total values to obtain average
    print ("The Mean Square Error is: ", MSE)
    return MSE

def MAE(inputs, target):
    summation = 0
    n = len(input)  # finding total number of items in list
    for i in range(0, n):  # looping through each element of the list
        difference = target[i] - inputs[i]
        abs_difference = abs(difference)
        summation += abs_difference 
    MAE= summation/n 
    print("The Mean Absolute Error is: ", MAE)
    return MAE