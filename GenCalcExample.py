
#! handles all the neural net data generation
import random
import os

#* Takes in a max value n and generates equations from 1-n
def generateNeuralNetDataCSV(n, name):
    #* Number of equations to generate 
    num = 10
    bounds = 20
    #* for loop
    with open(name + ".txt", "x") as f:
        for i in range(1, n + 1):
            for j in range(1, num + 1):
                f.write(str(i) + ',' + generateEquation(i,random.randint(2,j+2),bounds) + '\n')  
    return

def generateNeuralNetDataFileStructure(n, name):
    #* Number of equations to generate 
    num = 25000
    bounds = 100
    #* for loop
    direct = "F:/"+name
    os.mkdir(direct)
    for i in range(1, n + 1):
        path = direct + "/" + str(i)
        os.mkdir(path)
        path += "/"
        for j in range(1, num + 1):
            with open(path + str(j) + ".txt", "x") as f:
                f.write(generateEquation(i,random.randint(2,(j%10)+2),bounds))    
    return

#* Takes in a value and n, spits out a set of n equations with solution value
# (The work horse function)
def generateEquation(answer, l, b):
    # TODO: Fill Out Function
    # Rules - Needs to follow PEMDAS
    # Take a stack
    # generate a random +-*/ equation
    # if you can't do it, mark it as "bad" & pick a new one

    currentNumber = answer
    stack = []
    indexToSymbol = {0:'+',1:'-',2:'*',3:'/'}
    invertMapping = {'-':'+','+':'-','/':'*','*':'/'}
    for i in range(1,l) :
        attemptedSymbols = [0, 1, 2, 3]
        while len(attemptedSymbols) != 0:
            n = random.randint(0,len(attemptedSymbols) - 1)
            nextSymbol = attemptedSymbols[n]
            
            nextNum = 0
            if indexToSymbol[nextSymbol] == '+':
                nextNum = getNumberForPlus(currentNumber)
            elif indexToSymbol[nextSymbol] == '-':
                nextNum = getNumberForMinus(currentNumber)
            elif indexToSymbol[nextSymbol] == '*':
                nextNum = getNumberForMult(currentNumber)
            else:
                nextNum = getNumberForDiv(currentNumber)
            
            if nextNum == None:
                attemptedSymbols.remove(nextSymbol)
                continue
            total = eval(str(currentNumber) + " " + invertMapping[indexToSymbol[nextSymbol]] + " " + str(nextNum))
            if total > b or total < -b:
                continue
            else:
                attemptedSymbols.remove(nextSymbol)
            stack.insert(0,str(nextNum))
            stack.insert(0,str(indexToSymbol[nextSymbol]))
            currentNumber = total#eval(str(currentNumber) + " " + invertMapping[indexToSymbol[nextSymbol]] + " " + str(nextNum) + "\n")
            break

    stack.insert(0, str(currentNumber))

    return stringify(stack)

def stringify(stack):
    output = ""
    numParn = 0
    for n in range(len(stack)-1,-1,-1):
        if stack[n] == '/' or stack[n] == '*' :
            stack.insert(n, ')')
            numParn += 1
    for n in range(0, numParn):
        stack.insert(0, '(')
    return ' '.join(stack)

def getNumberForPlus(prevNum):
    return random.randint(0,20)

def getNumberForMinus(prevNum):
    return random.randint(0,20)

def getNumberForDiv(prevNum):
    return random.randint(1,10)

def getNumberForMult(prevNum):
    #implement Quadratic sieve
    #https://en.wikipedia.org/wiki/Quadratic_sieve
    return None

def main():
    anw = 10
    generateNeuralNetDataFileStructure(anw, "train")
    generateNeuralNetDataFileStructure(anw, "validate")
    generateNeuralNetDataFileStructure(anw, "test")

if __name__ == "__main__":
    main()
