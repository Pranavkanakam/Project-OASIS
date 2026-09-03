Things to improve upon:
1. Error handling is not there
2. In coord descent algo, I used MatrixXf and VectorXf, is it possible to use Matrix<float, ...> instead? 
3. I used a simple cyclical coordinate descent, but to generalize better we can randomized or "effectively" 
    cyclical coordinate descent where every cycle is a random permutation of (1,2,...P)
4. I am using static_cast<int>(..) everywhere to find N and P. Instead we can have do a const reference to pass the values around
5. We can trade space and optimize a bit more for time



Using the naive algorithm (with a lot of for loops):
for these hyperparameters: perform_CD_naive_on(W, X, y, 1e4, 1e-3, 0.05, 0.7);
I got (value in the bracket are the actual coefficient values):
0.533914 (0.5)
 2.85489 (3)
       0 (0)
(also, these values were obtained for the given tolerance in just 2 full cycles of coordinate updates -> so that's 4 (explicit) loops!)

So, as we can see, this soft-threshold is key for setting fellows to 0 (experimentally, yet to be proved mathematically)

Now, this is all the results I got for N = 100, P = 2, which is not the situtiation for which this algorithm was designed for. 
So let's try some insane stuff and see what happens.

Before that, I'd like to update my algorithm so that it runs better.
I will also chnage my data generation code so that I can easily set and change the number of parameters.
