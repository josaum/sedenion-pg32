import sympy as sp, itertools, sys
a=sp.symbols('a1:8'); b=sp.symbols('b1:8'); c=sp.symbols('c1:8')
A=(0,)+a; B=(0,)+b; C=(0,)+c
def conj(x): return (x[0],)+tuple(-t for t in x[1:])
def omul(x,y):
    x0,x1,x2,x3,x4,x5,x6,x7=x; y0,y1,y2,y3,y4,y5,y6,y7=y
    return (x0*y0-x1*y1-x2*y2-x3*y3-x4*y4-x5*y5-x6*y6-x7*y7,
     x0*y1+x1*y0+x2*y3-x3*y2+x4*y5-x5*y4-x6*y7+x7*y6,
     x0*y2-x1*y3+x2*y0+x3*y1+x4*y6+x5*y7-x6*y4-x7*y5,
     x0*y3+x1*y2-x2*y1+x3*y0+x4*y7-x5*y6+x6*y5-x7*y4,
     x0*y4-x1*y5-x2*y6-x3*y7+x4*y0+x5*y1+x6*y2+x7*y3,
     x0*y5+x1*y4-x2*y7+x3*y6-x4*y1+x5*y0-x6*y3+x7*y2,
     x0*y6+x1*y7+x2*y4-x3*y5-x4*y2+x5*y3+x6*y0-x7*y1,
     x0*y7-x1*y6+x2*y5+x3*y4-x4*y3-x5*y2+x6*y1+x7*y0)
def dot(x,y): return sum(p*q for p,q in zip(x,y))
def sub(x,y): return tuple(p-q for p,q in zip(x,y))
def add(x,y): return tuple(p+q for p,q in zip(x,y))
def smul(u,v): return (sub(omul(u[0],v[0]),omul(conj(v[1]),u[1])), add(omul(v[1],u[0]),omul(u[1],conj(v[0]))))
D=omul(omul(A,C),B)
P1,P2=smul((A,B),(C,D))
targets=[sp.expand(t) for t in P1+P2]
hyps={'haa':dot(A,A)-1,'hbb':dot(B,B)-1,'hab':dot(A,B),'hca':dot(C,A),'hcb':dot(C,B),'hcab':dot(C,omul(A,B))}
vars_=list(a)+list(b)+list(c)
lin=[sp.Integer(1)]+vars_
quad=lin+[vars_[i]*vars_[j] for i in range(len(vars_)) for j in range(i,len(vars_))]
# unknown coefficients: deg-2 hyps get linear coeff, hcab gets constant coeff
unk=[]; expr=0
for name,h in hyps.items():
    if name=='hcab':
        for i,m in enumerate(lin):
            s=sp.Symbol(f'k_{name}_{i}'); unk.append(s); expr+=s*m*h
    else:
        for i,m in enumerate(quad):
            s=sp.Symbol(f'k_{name}_{i}'); unk.append(s); expr+=s*m*h
expr=sp.expand(expr)
out=[]
for idx,t in enumerate(targets):
    if t==0: out.append(None); print(idx,"trivially 0"); continue
    diff=sp.Poly(expr-t,*vars_)
    eqs=diff.coeffs()
    sol=sp.linsolve(eqs,unk)
    if not sol: print(idx,"NO CERT"); sys.exit(1)
    sol=list(sol)[0]
    sol=[s.subs({u:0 for u in unk}) for s in sol]  # free params -> 0
    terms={}
    for u,val in zip(unk,sol):
        if val==0: continue
        name=u.name.split('_')[1]; i=int(u.name.split('_')[2])
        m = lin[i] if name=='hcab' else quad[i]
        terms[name]=terms.get(name,0)+val*m
    # verify
    chk=sp.expand(sum(coef*hyps[n] for n,coef in terms.items())-t)
    assert chk==0
    s=' + '.join(f'({sp.sstr(coef)}) * {n}' for n,coef in terms.items())
    out.append(s); print(idx, s)
import json; json.dump(out,open('certs.json','w'))
