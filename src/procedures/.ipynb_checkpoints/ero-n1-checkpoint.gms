$funclibin stolib stodclib
Functions normcdf     / stolib.cdfNormal /
          normicdf    / stolib.icdfNormal / ;

option nlp=conopt;

Scalar
    muA        "alternative mean"      / 2.0 /
    mu0        "null mean"             / 0.0 /
    p0         "prior of the null"     / 0.5 /
    sigma      "std dev of data"       / 1.0 /
    mfdr_alpha "mFDR control level" / 0.05 /
    phi        "cost of the test"   / 0.00475 / ;
    
    
Variables
    alpha "level of test"
    psi   "reward for successful rejection" ;

Variables
    rho     "power of the test"
    er1     "expected reward for test";
    
Positive Variables  alpha, psi, rho;

* Set initial values of alpha and rho
alpha.l = 0.001;
rho.l = 0.2;

alpha.lo = 1e-10;
alpha.up = 0.9;

Equations
    obj    "objective function"
    power  "power constraint"
    ero_c  "ero constraint"
    mfdr1  "mfdr control constraint 1"
    mfdr2  "mfdr control constraint 2";


obj..      er1     =e= psi ;
mfdr1..    psi     =l= phi/rho + mfdr_alpha ;
mfdr2..    psi     =l= phi/alpha + mfdr_alpha - 1;
ero_c..    phi/rho =e= phi/alpha - 1;
power..    rho     =e= 1 - normcdf( ((mu0-muA)/sigma) + normicdf(1-alpha,0,1), 0, 1);

Model ero / obj mfdr1 mfdr2 ero_c power /;

solve ero using nlp maxmizing er1;

display alpha.l, psi.l, rho.l;
