Project which aim is to recognise which digit speaker says
We can divide our project for few parts:
1.  Creating the GMM models from all of the data we have
    a) Recordings of 10 digits spoken by 22 speakers -> 220 recordings
    b) Parametrize this recordings by creating MFCC matrices with 39 features (13 normal features + delta + delta delta)
        -> 220 MFCC matrices
    c) Creating only 10 matrices for each digit concatenating matrices for each speaker
    d) Making 10 GMM models out of each MFCC matrix
2.  Validate the models by doing the cross validation - 11 tests, 2 speakers in each test group
    Recognition ratio after cross validation - 100%
3.  Doing the optimization tests to find the best parameters in both MFCC algorithm and GMM
4.  Evaluation of the data which has ben never used before to check how models are efficient
