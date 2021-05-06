import numpy as np
import random

class Perceptron():
    
    def __init__ (self, DataLength):
        
        self.Weights = np.zeros(DataLength)
        self.Bias = 0
    
    def PerceptronTraining(self, TrainingData, Labels, DataLength, MaximumIterations):
        self.trainingMisses = np.zeros(MaximumIterations)
        for iteration in range(MaximumIterations):
            for row in range(DataLength):
                
                self.Activation  = self.Bias + np.inner(TrainingData[row], self.Weights)
                if self.Activation > 0:
                    self.Activation = 1
                else:
                    self.Activation = -1
                #print(str(TrainingData[row]) + " predicted : " + str(self.Activation) + " Actual: " + str(Labels[row]))
                
                if (Labels[row]*self.Activation) <= 0:
                    self.trainingMisses[iteration] += 1
                    for weight in range(4):
                        self.Weights[weight] = self.Weights[weight] + (Labels[row]*TrainingData[row,weight])
                        
                    self.Bias = self.Bias + Labels[row]
            print ("Iteration:  " + str(iteration) + " , Accuracy: " + str((DataLength-self.trainingMisses[iteration])/DataLength*100))

        print("")
        print("--------------------------------------------------------")
        print ("Bias: " + str(self.Bias))
        print ("Weights: " + str(self.Weights))
        
    def Test(self,TestData): 
        return (self.Bias + np.inner(TestData, self.Weights))
    
    

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

trainingData = np.zeros((len(trainingFileShuffle), 4))
trainingOutputs = np.zeros((3,len(trainingFileShuffle)))

count = 0
for iteration in range(len(trainingFileShuffle)):
    row = trainingFileShuffle[iteration].split(',')
    classString = row[4]
    Output = classString[-2]
    Input = row[:4]
    if(float(Output) == 1):
        trainingData[shufflePositions[iteration]] = Input
        trainingOutputs[0, shufflePositions[iteration]] = 1
        trainingOutputs[1, shufflePositions[iteration]] = -1
        trainingOutputs[2, shufflePositions[iteration]] = -1
    elif(float(Output) == 2):
        trainingData[shufflePositions[iteration]] = Input
        trainingOutputs[0, shufflePositions[iteration]] = -1
        trainingOutputs[1, shufflePositions[iteration]] = 1
        trainingOutputs[2, shufflePositions[iteration]] = -1
    elif(float(Output) == 3):
        trainingData[shufflePositions[iteration]] = Input
        trainingOutputs[0, shufflePositions[iteration]] = -1
        trainingOutputs[1, shufflePositions[iteration]] = -1
        trainingOutputs[2, shufflePositions[iteration]] = 1


#print(trainingData)
#print(trainingOutputs)

#Testing
        
testingData = np.zeros((len(testingFile), 4))
testingOutputs = np.zeros(len(testingFile))

count = 0
for iteration in range(len(testingFileShuffle)):
    row = testingFileShuffle[iteration].split(',')
    classString = row[4]
    Output = classString[-2]
    Input = row[:4]
    testingData[iteration] = Input
    testingOutputs[iteration] = Output


    
#print(testingData)
#print(testingOutputs)


Perceptron1 = Perceptron(4)
Perceptron2 = Perceptron(4)
Perceptron3 = Perceptron(4)


Perceptron1.PerceptronTraining(trainingData, trainingOutputs[0], len(trainingFileShuffle), 20)
Perceptron2.PerceptronTraining(trainingData, trainingOutputs[1], len(trainingFileShuffle), 20)
Perceptron3.PerceptronTraining(trainingData, trainingOutputs[2], len(trainingFileShuffle), 20)

def Classify(TestData):
    perceptronActivations = np.zeros(3)
    perceptronActivations[0] = Perceptron1.Test(TestData)
    perceptronActivations[1] = Perceptron2.Test(TestData)
    perceptronActivations[2] = Perceptron3.Test(TestData)
    
    predictedClass = np.argmax(perceptronActivations) + 1
    return predictedClass

totalCorrect = 0
for iteration in range(len(testingOutputs)):
    prediction = Classify(testingData[iteration])
    print(str(testingData[iteration]) + " predicted : " + str(prediction) + " Actual: " + str(testingOutputs[iteration]))
    if(prediction == testingOutputs[iteration]):
        totalCorrect += 1
print("")       
print("Accuracy: " + str(totalCorrect/len(testingOutputs)*100))
    

#binaryPerceptron.Classify(testingData, testingOutputs, len(testingOutputs))
        
                    