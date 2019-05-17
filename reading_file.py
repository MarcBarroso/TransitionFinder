"""
Prototype of program that reads the file produced by 'transitionNfields' and recovers the arrays of the parameters.
Authors: Daan Verweij, Marc Barroso.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

file_name = '2fields_1str.txt'

if __name__ == '__main__':
    start_time = time.time()
    f = open(file_name, 'r')
    f.readline() #We read and disregard the first line, since it only contains the structure of the file
    f1 = f.readlines() #We save the other lines
    x = np.array([[0,0,0,0]])
    for i in range(len(f1)):
        if i % 2 == 1: #we visit only the odd lines, since we know that there is '-'*90 between them
            tmp = f1[i].split(' ') #we split each line in sections separated by ' '
            x = np.concatenate((x, [[float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3])]]), axis=0) #and save the numbers in the array
    x = np.delete(x, (0), axis=0)

    print x[np.where(x[:,0]==np.amax(x[:,0]))] #show me x where the value of the column is equal to the maximum value of that column. It might show more than just one entry
    print(" --- %s seconds ---" % (time.time() - start_time))
    plt.plot(x[:,0], x[:,3], 'ro')
    plt.show()

