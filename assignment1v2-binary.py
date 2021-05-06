import numpy as np
import random

class Perceptron():
    
    def __init__ (self, DataLength):
        
        self.Weights = np.zeros(DataLength)
        self.Bias = 0
    
    def PerceptronTraining(self, TrainingData, Labels, DataLength, MaximumIterations):
        trainingMisses = np.zeros(MaximumIterations)
        for iteration in range(MaximumIterations):
            for row in range(DataLength):
                
                Activation  = self.Bias + np.inner(TrainingData[row], self.Weights)
                if Activation > 0:
                    Activation = 1
                else:
                    Activation = -1
                #print(str(TrainingData[row]) + " predicted : " + str(Activation) + " Actual: " + str(Labels[row]))
                
                if (Labels[row]*Activation) <= 0:
                    trainingMisses[iteration] += 1
                    for weight in range(4):
                        self.Weights[weight] = self.Weights[weight] + (Labels[row]*TrainingData[row,weight])
                        
                    self.Bias = self.Bias + Labels[row]
            print ("Iteration:  " + str(iteration) + " , Accuracy: " + str((DataLength-trainingMisses[iteration])/DataLength*100))

        print("")
        print("--------------------------------------------------------")
        print ("Bias: " + str(self.Bias))
        print ("Weights: " + str(self.Weights))
    
    def Classify(self,TestData, Labels, DataLength): 
        testingMisses = 0
        for row in range(DataLength):
                
                Activation  = self.Bias + np.inner(TestData[row], self.Weights)
                if Activation > 0:
                    Activation = 1
                else:
                    Activation = -1
                    
                if (Labels[row]*Activation) <= 0:
                    testingMisses += 1
                    

        print("Accuracy: ")
        print((DataLength-testingMisses)/DataLength*100)
        

        
    
    
#Change this line to specify which classes to test between
classes = [1,3]

fileObj = open("train.data", "r")
trainingFileShuffle = fileObj.readlines()
#trainingFileShuffle = random.sample(trainingFile, len(trainingFile))
fileObj.close()


fileObj = open("test.data", "r")
testingFile = fileObj.readlines()
testingFileShuffle = random.sample(testingFile, len(testingFile))
fileObj.close()



#Training

#generate new values
#shufflePositions = list(range(0,len(trainingFileShuffle)))
#shufflePositions = random.sample(shufflePositions, len(shufflePositions))
#print(shufflePositions)

shufflePositions = [114, 113, 92, 13, 10, 83, 20, 82, 77, 47, 87, 119, 54, 105, 1, 102, 103, 48, 99, 63, 21, 53, 41, 25, 45, 80, 91, 98, 0, 88, 12, 2, 84, 17, 3, 76, 94, 111, 93, 110, 101, 74, 52, 38, 56, 43, 71, 97, 14, 107, 62, 100, 118, 50, 72, 9, 4, 73, 30, 46, 49, 8, 6, 108, 70, 66, 7, 18, 59, 112, 90, 115, 19, 81, 79, 42, 89, 64, 24, 86, 37, 36, 32, 61, 22, 15, 109, 69, 39, 65, 78, 16, 55, 51, 28, 33, 68, 60, 104, 117, 116, 40, 106, 23, 85, 95, 96, 27, 11, 31, 35, 44, 26, 29, 58, 75, 5, 34, 57, 67]



#Training
count = 0
for iteration in range(len(trainingFileShuffle)):
    row = trainingFileShuffle[shufflePositions[iteration]].split(',')
    classString = row[4]
    Output = classString[-2]
    Input = row[:4]
    if(float(Output) == classes[0]):
        count +=1
    elif(float(Output) == classes[1]):
        count+=1
        

trainingData = np.zeros((count, 4))
trainingOutputs = np.zeros(count)

count = 0
for iteration in range(len(trainingFileShuffle)):
    row = trainingFileShuffle[shufflePositions[iteration]].split(',')
    classString = row[4]
    Output = classString[-2]
    Input = row[:4]
    if(float(Output) == classes[0]):
        trainingData[count] = Input
        trainingOutputs[count] = 1
        count += 1
    elif (float(Output) == classes[1]):
        trainingData[count] = Input
        trainingOutputs[count] = -1
        count += 1


#print(trainingData)
#print(trainingOutputs)

#Testing
count = 0
for iteration in range(len(testingFileShuffle)):
    row = testingFileShuffle[iteration].split(',')
    classString = row[4]
    Output = classString[-2]
    Input = row[:4]
    if(float(Output) == classes[0]):
        count +=1
    elif(float(Output) == classes[1]):
        count+=1
        

testingData = np.zeros((count, 4))
testingOutputs = np.zeros(count)

count = 0
for iteration in range(len(testingFileShuffle)):
    row = testingFileShuffle[iteration].split(',')
    classString = row[4]
    Output = classString[-2]
    Input = row[:4]
    if(float(Output) == classes[0]):
        testingData[count] = Input
        testingOutputs[count] = 1
        count += 1
    elif (float(Output) == classes[1]):
        testingData[count] = Input
        testingOutputs[count] = -1
        count += 1

    
#print(testingData)
#print(testingOutputs)


print ("Classes: " + str(classes))
print("")
binaryPerceptron = Perceptron(4)
binaryPerceptron.PerceptronTraining(trainingData, trainingOutputs, len(trainingOutputs), 20)
binaryPerceptron.Classify(testingData, testingOutputs, len(testingOutputs))
        
                    