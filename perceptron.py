import random as random
import math as math


class perceptron:
    def __init__(self):
        # Parameters for our perceptron
        self.MAX_ITERATION = 50
        self.LEARNING_RATE = 0.1
        self.NUM_INSTANCES = 5
        self.THETA = 0

        # 3 weights: 1 is bias and 2 in list
        self.BIAS = -1
        self.weightslist = []

        # 2 types of error
        self.localError = 0.1
        self.globalError = 0.1

        # 1 output
        self.outputlist = []

        # correct times when we final test
        self.correct = 0

        # original numbers
        even0 = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]
        odd1 = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]
        even2 = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
        odd3 = [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0]
        even4 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0]

        # initial value for list, with x1,x2...x25 all in it
        self.xlist = []

        for n in range(0, 25, 1):
            self.xlist.append(even0)
            self.xlist.append(odd1)
            self.xlist.append(even2)
            self.xlist.append(odd3)
            self.xlist.append(even4)

        # inital outputlist, the order is even then odd, so should be 00011
        self.outputlist = [0, 1, 0, 1, 0]

        # initial weights
        self.weightslist = self.__randomnum(-1, 1, 25)

    def start(self):
        print("\n@@@@@@@@@@@@@@@@@@@@@@@\n\nSTAGE 1: Computing the weights!\n")
        self.__stage1_computing_the_weights()
        print("\n@@@@@@@@@@@@@@@@@@@@@@@\n\nSTAGE 2: Print the equation!")
        self.__stage2_print_the_equation()
        print("\n@@@@@@@@@@@@@@@@@@@@@@@\n\nSTAGE 3: Use the equation to test!")
        self.__stage3_test()

    def __stage1_computing_the_weights(self):
        iteration = 0

        # START COMPUTING THE WEIGHTS
        while self.globalError != 0 and iteration <= self.MAX_ITERATION:
            iteration += 1
            self.globalError = 0

            # loop through all instances (complete one epoch)
            for i in range(0, 5, 1):
                # calculate predicted class
                output = self.__computeoutput(self.weightslist, i)

                # differences between predicted and actual class value
                self.localError = self.outputlist[i] - output

                # update weights and bias
                for n in range(0, 25, 1):
                    self.weightslist[n] += self.LEARNING_RATE * self.localError * self.xlist[i][n]
                self.BIAS += self.LEARNING_RATE * self.localError * (-1)

                # sum of squared error(error value of all instances
                self.globalError += (self.localError * self.localError)

            # print the results, Root mean squared error
            print("Iteration:", iteration, " RMSE:", math.sqrt(self.globalError / self.NUM_INSTANCES))

    def __stage2_print_the_equation(self):
        print("\nDecision boundary equation:")
        print(self.weightslist[0], "* x1 +", self.weightslist[1], "* x2 +\n", \
              self.weightslist[2], "* x3 +", self.weightslist[3], "* x4 +\n", \
              self.weightslist[4], "* x5 +", self.weightslist[5], "* x6 +\n", \
              self.weightslist[6], "* x7 +", self.weightslist[7], "* x8 +\n", \
              self.weightslist[8], "* x9 +", self.weightslist[9], "* x10 +\n", \
              self.weightslist[10], "* x11 +", self.weightslist[11], "* x12 +\n", \
              self.weightslist[12], "* x13 +", self.weightslist[13], "* x14 +\n", \
              self.weightslist[14], "* x15 +", self.weightslist[15], "* x16 +\n", \
              self.weightslist[16], "* x17 +", self.weightslist[17], "* x18 +\n", \
              self.weightslist[18], "* x19 +", self.weightslist[19], "* x20 +\n", \
              self.weightslist[20], "* x21 +", self.weightslist[21], "* x22 +\n", \
              self.weightslist[22], "* x23 +", self.weightslist[23], "* x24 +\n", \
              self.weightslist[24], "* x25 +", self.BIAS, "= 0")

    def __stage3_test(self):
        # START TESTING NOW, GOOD LUCK!
        # generate test number, simply flip the bits
        newlist = []
        for innerlist in self.xlist:
            newlist.append(self.__flipbits(innerlist))
        self.xlist = newlist

        # we will do a loop here, stop until it guess right 5 times
        # We choose 5 because it is the quantity of our test data, 5 numbers
        roundcounting = 0
        while self.correct != 5:
            # check their classification 01010
            for i in range(0, 5, 1):
                roundcounting += 1
                output = self.__computeoutput(self.weightslist, i)
                print("\n====\n");
                print("Round", roundcounting, ":")
                print("The number is", i)
                if self.outputlist[i] == 0:
                    print("It should be EVEN")
                else:
                    print("It should be ODD")

                if output == 0:
                    print("I assume it's EVEN")
                else:
                    print("I assume it's ODD")

                if output == self.outputlist[i]:
                    self.correct += 1
                    print("Guess right for", self.correct, "times")
                    if self.correct == 5:
                        break
                    print(5 - self.correct, " more times to quit.")
                else:
                    print("I'm wrong, guess again!")

        print("\n@@@@@@@@@@@@@@@@@@@@@@@\n\nALL OVER, we have right for 5 times.")
        print("We used", roundcounting, "rounds to achieve it.\n")
        print("You know, it's not fair!")
        print("You told me to use 1 to distinguish the number, \nbut you change the rule!\n")
        print("And I can guess it right even for that!\nSuch a intelligent AI I am!\nAHAHAHAHAHAHAHAHAHAHAHA")

    def __computeoutput(self, weightlist, i):
        mysum = self.xlist[i][0] * weightlist[0] + \
                self.xlist[i][1] * weightlist[1] + self.xlist[i][2] * weightlist[2] + \
                self.xlist[i][3] * weightlist[3] + self.xlist[i][4] * weightlist[4] + \
                self.xlist[i][4] * weightlist[5] + self.xlist[i][6] * weightlist[6] + \
                self.xlist[i][5] * weightlist[7] + self.xlist[i][8] * weightlist[8] + \
                self.xlist[i][9] * weightlist[9] + self.xlist[i][10] * weightlist[10] + \
                self.xlist[i][11] * weightlist[11] + self.xlist[i][12] * weightlist[12] + \
                self.xlist[i][13] * weightlist[13] + self.xlist[i][14] * weightlist[14] + \
                self.xlist[i][15] * weightlist[15] + self.xlist[i][16] * weightlist[16] + \
                self.xlist[i][17] * weightlist[17] + self.xlist[i][18] * weightlist[18] + \
                self.xlist[i][19] * weightlist[19] + self.xlist[i][20] * weightlist[20] + \
                self.xlist[i][21] * weightlist[21] + self.xlist[i][22] * weightlist[22] + \
                self.xlist[i][23] * weightlist[23] + self.xlist[i][24] * weightlist[24] + self.BIAS

        if mysum >= self.THETA:
            return 1
        else:
            return 0

    def __randomnum(self, mymin, mymax, quantity):
        if quantity == 1:
            randomnum = random.uniform(0, 1)
            return randomnum
        else:
            randomlist = []
            for x in range(0, quantity, 1):
                randomlist.append(random.uniform(mymin, mymax))
            return randomlist

    def __addoutput(self, value, quantity):
        mylist = []
        for x in range(0, quantity, 1):
            mylist.append(value)
        return mylist

    def __flipbits(self, cominglist):
        result = cominglist
        randomindex = random.sample(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 5)
        for num in randomindex:
            if cominglist[num] == 0:
                result[num] = 1
            elif cominglist[num] == 1:
                result[num] = 0
        return result


test = perceptron()
test.start()
