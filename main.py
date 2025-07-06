# Portfolio Optimization using Mean Variance
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from scipy import optimize

def MaximiseReturns(MeanReturns, PortfolioSize):
    c = (np.multiply(-1, MeanReturns))
    A_eq = np.ones([1, PortfolioSize])
    b_eq = [1]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0,1), method='simplex')
    return res

def MinimizeRisks(CovarReturns, PortfolioSize):
    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def ConstraintEq(x):
        A = np.ones(x.shape)
        b = 1
        constraintVal = np.matmul(A, x.T) - b
        return constraintVal

    x_initial = np.repeat(0.1, PortfolioSize)
    cons = ({'type':'eq','fun': ConstraintEq})
    lower_bound = 0
    upper_bound = 1
    bounds = tuple([(lower_bound, upper_bound) for x in x_initial])

    opt = optimize.minimize(f, x0=x_initial, args=(CovarReturns,), bounds=bounds,
                           constraints=cons, tol=10**-3)
    return opt


def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):

    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        Aeq = np.ones(x.shape)
        bEq = 1
        EqconstraintVal = np.matmul(Aeq, x.T) - bEq
        return EqconstraintVal

    def constraintIneq(x, MeanReturns, R):
        AIneq = np.array(MeanReturns)
        bIneq = R
        IneqconstraintVal = np.matmul(AIneq, x.T) - bIneq
        return IneqconstraintVal  # Fixed typo here too

    x_initial = np.repeat(0.1, PortfolioSize)

    cons = (
        {'type': 'eq', 'fun': constraintEq},
        {'type': 'ineq', 'fun': constraintIneq, 'args': (MeanReturns, R)}
    )

    lb = 0
    ub = 1

    bnds = tuple([(lb, ub) for _ in x_initial])

    opt = optimize.minimize(
        f, args=(CovarReturns,), method='trust-constr',
        x0=x_initial, bounds=bnds, constraints=cons, tol=10**-3
    )

    return opt



# Function computes asset returns
def StockReturnsComputing(StockPrice, Rows, Columns):
    
    
    
    StockReturn = np.zeros([Rows-1, Columns])
    for j in range(Columns):        # j: Assets
        for i in range(Rows-1):     # i: Daily Prices
            StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100

    return StockReturn

df = pd.read_csv('DJIA_Apr112014_Apr112019_kpf1.csv')

#print(df.head())
#print(df.shape) 
portfolio_size = 15
Columns = portfolio_size

assetLabels = df.columns[1:Columns+1].tolist() #just converts it to a list
print(assetLabels)

StockData = df.iloc[0:, 1:]
StockPrice = StockData.to_numpy()  #converts the dataframe to a numpy array
print(StockPrice)

[Rows, Cols] = StockPrice.shape
print(Rows, Cols)

Stock_Return = StockReturnsComputing(StockPrice, Rows, Cols)

meanReturns = Stock_Return.mean(axis=0)
covReturns = np.cov(Stock_Return, rowvar=False)

np.set_printoptions(precision=3, suppress = True)
#print('Mean returns of assets in k-portfolio 1\n', meanReturns)
#print('Covariance of returns of assets in k-portfolio 1\n', covReturns)


#Maximal expected portfolio return computation for the k-portfolio
result1 = MaximiseReturns(meanReturns, portfolio_size)
maxReturnWeights = result1.x
maxExpPortfolioReturn = np.matmul(meanReturns.T, maxReturnWeights)
print("Maximal Expected Portfolio Return:   %7.4f" % maxExpPortfolioReturn )

#expected portfolio return computation for the minimum risk k-portfolio 
result2 = MinimizeRisks(covReturns, portfolio_size)
minRiskWeights = result2.x
minRiskExpPortfolioReturn = np.matmul(meanReturns.T, minRiskWeights)
print("Expected Return of Minimum Risk Portfolio:  %7.4f" % minRiskExpPortfolioReturn)

low = minRiskExpPortfolioReturn
high = maxExpPortfolioReturn
xOptimal = []
minRiskPoint = []
expPortfolioReturnPoint = []
increment = 0.001



while (low < high):
    result3 = MinimizeRiskConstr(meanReturns, covReturns, portfolio_size, low)
    xOptimal.append(result3.x)
    expPortfolioReturnPoint.append(low)
    low = low + increment
    xOptimalArray = np.array(xOptimal)

minRiskPoint = np.diagonal(np.matmul((np.matmul(xOptimalArray,covReturns)),\
                                     np.transpose(xOptimalArray)))
riskPoint =   np.sqrt(minRiskPoint*251) 

#obtain expected portfolio annualized return for the 
#efficient set portfolios, for trading days = 251
retPoint = 251*np.array(expPortfolioReturnPoint) 
#display efficient set portfolio parameters
print("Size of the  efficient set:", xOptimalArray.shape )
print("Optimal weights of the efficient set portfolios: \n", xOptimalArray)
print("Annualized Risk and Return of the efficient set portfolios: \n", \
                                                np.c_[riskPoint, retPoint])

#Graph Efficient Frontier
import matplotlib.pyplot as plt

NoPoints = riskPoint.size

colours = "blue"
area = np.pi*3

plt.title('Efficient Frontier for k-portfolio 1 of Dow stocks')
plt.xlabel('Annualized Risk(%)')
plt.ylabel('Annualized Expected Portfolio Return(%)' )
plt.scatter(riskPoint, retPoint, s=area, c=colours, alpha =0.5)
plt.show()




