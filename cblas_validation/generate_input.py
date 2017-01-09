
import os
import sys 
import random

inputA = 'A_col_major.txt'
inputB = 'B_col_major.txt'



def write_to_file(size, init, inputFile):
    for i in range(0,size):
        A = []
        for j in range(0, size):
            if  init == '1':        
                value = 1
            elif init == 'r':
                value = random.randint(0, 10)
            elif init == 'i':
                value = i*size+j
            elif init == '0':        
                value = 0
            A.append(value)
        #print A
        inputFile.writelines( "%d " % item for item in A )
        inputFile.write("\n")        





#N is the matrix size, change N 
N = 192

inputA = open(inputA, "w")
#options for the second argument for initializaton
#   '1' : set every number with 1
#   '0' : set every number with 0
#   'r' : set with random number [1, 10]
#   'i' : set each to its index in the matrix 
write_to_file(N, 'r', inputA)

inputB = open(inputB, "w")
write_to_file(N, '1', inputB)

#perferred to be '0'
#inputC = open(inputC, "w") 
#write_to_file(64, '0', inputC)

inputA.close()
inputB.close()
#inputC.close()
