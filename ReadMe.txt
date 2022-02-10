In order to run our program, we used Anaconda's Spyder IDE.   I suggest using this to test the program.

For our first question, the binary implementation of a perceptron, we have assignment1v2-binary.py   on line 58
the classes used for comparison can be modified (1,2 & 3 respectively for our data as input options) Once 
you have chosen the classes, simply running the program should give the output required.

For our multiclass  version of the perceptron we use assignment1v3-multiclass simply just run the python file.

Regarding the regularized version, whilst we had difficulty implementing this correctly, it has a co-efficient
which can be changed on line 119,   with 0.01 this works, but higher values destablise the model as I believe
I needed to normalise the weights and include stochastic gradient descent. That in earnest, simply running
the file should work here.
