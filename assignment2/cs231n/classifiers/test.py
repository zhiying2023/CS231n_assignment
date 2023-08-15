import numpy as np


def test1():
    test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(4, 3)
    train = np.array([5, 3, 4, 6, 1, 6]).reshape(2, 3)
    print(test, "\n", train)

    row, _ = test.shape
    column, _ = train.shape
    dists = np.zeros((row, column))
    for i in range(row):
        dists[i, :] = np.sqrt(np.square(test[i, :] - train).sum(axis=1))
    print(dists)

    dists = np.zeros((row, column))
    test_square = np.square(test).sum(axis=1).reshape(-1, 1)
    train_square = np.square(train).sum(axis=1).reshape(1, -1)
    dists = np.sqrt(test_square + train_square - 2 * test.dot(train.T))
    print(dists)


class test2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def p(self, k):
        if k == 1:
            print(x)
        else:
            print(y)


if __name__ == "__main__":
    N = 5
    D = 4
    C = 3
    x = np.random.randint(-10, 10, (2, 2, 2))
    y = np.random.randint(0, 2, (10,))

    # y=np.array([0,1,2,3,4]).reshape((-1,1))
    W = np.random.randint(1, 10, (4, 3)) / 10
    print("x", x, "\n")
    # print('y', y, '\n')
    # print('W', W, '\n')
    test = test2(x, y)
    k = 1
    test(k)
