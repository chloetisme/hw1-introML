import numpy
import os
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_EXERCISE_6 = True
RUN_EXERCISE_7 = True
RUN_EXERCISE_8 = True
RUN_EXERCISE_9 = True


# -----------------------------------------------------------------------------
# Global paths
# -----------------------------------------------------------------------------

DATA_ROOT = '../data'
PATH_TO_WALK_DATA = os.path.join(DATA_ROOT, 'walk.txt')
PATH_TO_X_DATA = os.path.join(DATA_ROOT, 'X.txt')
PATH_TO_W_DATA = os.path.join(DATA_ROOT, 'w.txt')

FIGURES_ROOT = '../figures'
PATH_TO_WALK_FIGURE = os.path.join(FIGURES_ROOT, 'walk.png')


# -----------------------------------------------------------------------------

def exercise_6(path_to_data, path_to_figure):
    '''
    
    Parameters
    ----------
    path_to_data : where the data is located.
    path_to_figure : where the data is located.

    Returns
    -------
    walk_arr : array created from walk data.
    walk_min : The minimum data point in walk array.
    walk_max : The maximum data point in walk array.
    walk_arr_scaled : The new scaled data set.

    '''

    print('='*30)
    print('Running exercise_6()')

    #### YOUR CODE HERE ####

    # loading text file of data to variable
    walk_arr = numpy.loadtxt(PATH_TO_WALK_DATA, delimiter=',')
    # print(walk_arr) # used for testing

    #### YOUR CODE HERE ####
    # plot the data using matplotlib plot!

    plt.plot(walk_arr)  # creates a plot with the data specified
    plt.xlabel("Step")  # labels x axis
    plt.ylabel("Location")  # labels y axis
    plt.title("Random Walk")  # Creates title over plot
    plt.savefig(PATH_TO_WALK_FIGURE)

    print(f'walk_arr.shape: {walk_arr.shape}')

    #### YOUR CODE HERE ####
    walk_min = numpy.min(walk_arr)

    print(f'walk_min: {walk_min}')

    #### YOUR CODE HERE ####
    walk_max = numpy.max(walk_arr)

    print(f'walk_max: {walk_max}')
    
    current_range = walk_max-walk_min 
    new_min = -2
    new_max = 3     
    new_range = new_max-new_min
    scale_factor = new_range/current_range
    shift_factor = new_min - scale_factor*walk_min
        
    
    #### YOUR CODE HERE ####
    #linearly scales data based on given "new" min/max parameters
    walk_arr_scaled = scale_factor*walk_arr + shift_factor
    plt.plot(walk_arr_scaled) 

    print('DONE exercise_6()')

    return walk_arr, walk_min, walk_max, walk_arr_scaled


# -----------------------------------------------------------------------------

def exercise_7():
    '''
    
    Parameters
    ----------
    no parameters entered

    Returns
    -------
    experiment_outcomes_1 : outcomes of first experiment
    experiment_outcomes_2 : outcomes of second experiment
    experiment_outcomes_3 : outcomes of third experiment

    ''' 
    
    print('=' * 30)
    print('Running exercise_7()')

    #### YOUR CODE HERE ####
    numpy.random.seed(7)

    # This determines how many times we "throw" the
    #   2 six-sided dice in an experiment
    num_dice_throws = 10000  # don't edit this! should be 10000

    # This determines how many trials in each experiment
    #   ... that is, how many times we'll throw our two
    #   6-sided dice num_dice_throws times
    num_trials = 10  # don't edit this!

    # Yes, you can have functions inside of functions!
    def run_experiment():
        
        
        '''
          Parameters
          ----------
          no parameters entered

          Returns
          -------
        trail outcomes: outcomes of trails

         '''
        
        trial_outcomes = list()
        for trial in range(num_trials): #I'm confused on what this line is doing, where does trail come from?
            #### YOUR CODE HERE ####
            # In the following, make it so that probability_estimate is an estimate
            # of the probability of throwing 'doubles' with two fair six-sided dice
            # (i.e., the probability that the dice end up with teh same values)
            # based on throwing the two dice num_dice_throws times.
            dice_same = 0 #this is for counting the number of times the dice "roll" the same number
            for roll in range(num_dice_throws):
                
                dice_1 = numpy.random.randint(6)
                #print(f'dice_1: {dice_1}') #for testing
                dice_2 = numpy.random.randint(6)
                #print(f'dice_2: {dice_2}') #for testing
                if dice_1 == dice_2:
                    dice_same = dice_same + 1
                    #print(f'dice_same: {dice_same}') #for testing
           
                probability_estimate = dice_same/num_dice_throws

            # Save the probability estimate for each trial (you don't need to change
            # this next line)
            trial_outcomes.append(probability_estimate)
        return trial_outcomes

    experiment_outcomes_1 = run_experiment()
    
    
    print(f'experiment_outcomes_1: {experiment_outcomes_1}')

    print(f'do it again!')

    experiment_outcomes_2 = run_experiment()
    print(f'experiment_outcomes_2: {experiment_outcomes_2}')

    print('Now reset the seed')

    #### YOUR CODE HERE ####
    numpy.random.seed(7) # reset the numpy random seed back to 7

    experiment_outcomes_3 = run_experiment()

    print(f'experiment_outcomes_3: {experiment_outcomes_3}')

    print("DONE exercise_7()")

    return experiment_outcomes_1, experiment_outcomes_2, experiment_outcomes_3

# -----------------------------------------------------------------------------

def exercise_8():
    """
    

    Returns
    -------
    x :  2-d array of random number of shape (3, 1)
    y :  2-d array of random number of shape (3, 1)
    v1 : sum of x and y
    v2 : element-wise product of x and y
    xT : Transpose of x
    v3 : dot product of x and y
    A : 2-d array of random numbers of shape (3, 3)
    v4 : dot product of x-transpose with A
    v5 : dot product of x-transpose with A and the product with y
    v6 : inverse of A
    v7 : dot product of A with its inverse. (should be near identity)

    """
    

    print("=" * 30)
    print("Running exercise_8()")

    numpy.random.seed(7) # set the numpy random seed to 7

    #### YOUR CODE HERE ####
    # Set x to a 2-d array of random number of shape (3, 1)
    x = numpy.random.rand(3, 1)

    print(f'x:\n{x}')

    #### YOUR CODE HERE ####
    # Set y to a 2-d array of random number of shape (3, 1)
    y = numpy.random.rand(3, 1)
    
    print(f'y:\n{y}')

    #### YOUR CODE HERE ####
    # Calculate the sum of x and y
    v1 = x + y

    print(f'v1:\n{v1}')

    #### YOUR CODE HERE ####
    # Calculate the element-wise product of x and y
    v2 = x*y

    print(f'v2:\n{v2}')

    #### YOUR CODE HERE ####
    # Transpose x
    xT = numpy.transpose(x)

    print(f'xT: {xT}')

    #### YOUR CODE HERE ####
    # Calculate the dot product of x and y
    v3 = numpy.dot(xT, y) #??? I think this is wrong

    print(f'v3: {v3}')

    #### YOUR CODE HERE ####
    # Set A to a 2-d array of random numbers of shape (3, 3)
    A = numpy.random.rand(3, 3)

    print(f'A:\n{A}')

    #### YOUR CODE HERE ####
    # Compute the dot product of x-transpose with A
    v4 = numpy.dot(xT, A)

    print(f'v4: {v4}')

    #### YOUR CODE HERE ####
    # Compute the dot product of x-transpose with A and the product with y
    v5 = numpy.dot(v4, y)

    print(f'v5: {v5}')

    #### YOUR CODE HERE ####
    # Compute the inverse of A
    v6 = numpy.linalg.inv(A)

    print(f'v6:\n{v6}')

    #### YOUR CODE HERE ####
    # Compute the dot product of A with its inverse.
    #   Should be near identity (save for some numerical error)
    v7 =  numpy.dot(A, v6)

    print(f'v7:\n{v7}')

    return x, y, v1, v2, xT, v3, A, v4, v5, v6, v7


# -----------------------------------------------------------------------------

def exercise_9(path_to_X_data, path_to_w_data):
    """
    
    Parameters
    ----------
    path_to_X_data : path to x_data
    path_to_w_data : path to w_data

    Returns
    -------
    X : x_data into array
    w : w_data into array
    x_n1 : first column of x data
    x_n2 : second column of x data.
    scalar_result : right-hand side of Exercise 3
    XX : X-transpose times X
    wXXw : final left-hand side value

    """
    

    print("="*30)
    print("Running exercise_9()")

    #### YOUR CODE HERE ####
    # load the X and w data from file into arrays
    X = numpy.loadtxt(PATH_TO_X_DATA, delimiter=',')
    w = numpy.loadtxt(PATH_TO_W_DATA, delimiter=',')

    print(f'X:\n{X}')
    print(f'w: {w}')

    #### YOUR CODE HERE ####
    # Extract the column 0 (x_n1) and column 1 (x_n2) vectors from X
    x_n1 = X[:, 0]
    x_n2 = X[:, 1]

    print(f'x_n1: {x_n1}')
    print(f'x_n2: {x_n2}')

    
    #### YOUR CODE HERE ####
    # Use scalar arithmetic to compute the right-hand side of Exercise 3
    #   (Exercise 1.3 from FCMA p.35)
    # Set the final value to
    
    x_n1_square= numpy.square(x_n1)
    x_n1_square_sum= sum(x_n1_square)
    w_0_square= numpy.square(w[0])
    a=w_0_square*x_n1_square_sum #the complete first term on the rhs
    
    x_n1_x_n2= x_n1*x_n2
    x_n1_x_n2_sum= sum(x_n1_x_n2)
    w_0_w_1= w[0]*w[1]
    b=2*w_0_w_1*x_n1_x_n2_sum #the complete first term on the rhs
    
    x_n2_square= numpy.square(x_n2)
    x_n2_square_sum= sum(x_n2_square)
    w_1_square= numpy.square(w[1])
    c=w_1_square*x_n2_square_sum #the complete third term on the rhs
    #print(f'test: {b}') #for testing
    
    scalar_result = a + b + c

    print(f'scalar_result: {scalar_result}')

    #### YOUR CODE HERE ####
    # Now you will compute the same result but using linear algebra operators.
    #   (i.e., the left-hand of the equation in Exercise 1.3 from FCMA p.35)
    # You can compute the values in any linear order you want (but remember,
    # linear algebra is *NOT* commutative!), however here will require you to
    # first computer the inner term: X-transpose times X (XX), and then
    # below you complete the computation by multiplying on the left and right
    # by w (wXXw)
    
    X_transpose=numpy.transpose(X)
    w_transpose=numpy.transpose(w)
    print(f'wtrans:\n{w_transpose}')
    XX = numpy.matmul(X_transpose, X)
    


    print(f'XX:\n{XX}')

    #### YOUR CODE HERE ####
    # Now you'll complete the computation by multiplying on the left and right
    # by w to determine the final value: wXXw
    w_transXX = numpy.matmul(w_transpose,XX)
    print(f'wtransXX:\n{w_transXX}')
    wXXw = numpy.matmul(w_transXX,w)

    print(f'wXXw: {wXXw}')

    print("DONE exercise_9()")

    return X, w, x_n1, x_n2, scalar_result, XX, wXXw


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_EXERCISE_6:
        exercise_6(PATH_TO_WALK_DATA, PATH_TO_WALK_FIGURE)
        plt.show() #don't know if this is working as intended?
    if RUN_EXERCISE_7:
        exercise_7()
    if RUN_EXERCISE_8:
        exercise_8()
    if RUN_EXERCISE_9:
        exercise_9(PATH_TO_X_DATA, PATH_TO_W_DATA)
