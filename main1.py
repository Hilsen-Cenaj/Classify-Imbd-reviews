from sympy.printing.tests.test_tensorflow import tf
from Load_data import load, call_train

if __name__ == '__main__':
    for i in range(100, 3600, 500):
        print("###Train our algorithm###")
        call_train(i)

    """
    
    calltrain(X_train,y_train)
    print("-----------Testing Data--------------")
    flag = False #means we have test data
    print("-----------Our code------------------")
    calltest(path_test_from_test, number, flag)
    print("-----------Compared code------------------")
    call(path_train, path_test_from_test, number, flag)
    print("-----------Training Data--------------")
    flag = True
    print("-----------Our code------------------")
    calltest(path_test_from_train, number, flag)
    print("-----------Compared code------------------")
    call(path_train, path_test_from_train, number, flag)
    """
