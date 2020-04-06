import pandas as pd
import numpy as numpy
import quadprog as quadprog
import cvxopt as cvxopt

yfile = pd.ExcelFile('Y_label.xlsx')
y=yfile.parse('Sheet1')
a,label=y.T.to_numpy()
xfile = pd.ExcelFile('X_data.xlsx')
x=xfile.parse('Sheet1')
id,id1,taskid,formula,density,oxide,sgnumber,bandgap,mag,lata,latb,latc,alpha,beta,gamma,vol,atom1,atom2,ratio1,ratio2,group1,group2,energy=x.T.to_numpy()
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

P=numpy.zeros((3,3))
for j in range (0,3):
    P[j,j]=1

q=numpy.zeros((3,1))
G=numpy.zeros((21016,3))
for k in range (0,21016):
    G[k,0]=-label[k]*alpha[k]
for o in range (0,21016):
    G[o,1]=-label[o]*beta[o]
for p in range (0,21016):
    G[p,2]=-label[p]*gamma[p]
h=numpy.zeros((21016,1))
for i in range (0,21016):
    h[i,0]=-1





quadprog.solve_qp(P,q,G,h,0)