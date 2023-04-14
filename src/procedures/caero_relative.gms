$funclibin stolib stodclib
Functions normcdf     / stolib.cdfNormal /
          normicdf    / stolib.icdfNormal / ;

option nlp=conopt;

Scalar
    muA        "alternative mean"      / 2.0 /
    mu0        "null mean"             / 0.0 /
    sigma      "std dev of data"       / 1.0 /
    q0         "prior of the null"     / 0.9 /
    mfdr_alpha "mFDR control level"    / 0.05 /
    Wd         "dollar wealth"         / 1000 /
    Wa         "alpha wealth"          / 0.0475 /
    c          "cost per sample"       / 1.0 /;
    
    
Variables
    alpha "level of test"
    psi   "reward for successful rejection" 
    n     "number of samples"
    phi   "cost of test";

Variables
    rho     "power of the test"
    er1     "expected reward for test";
    
Positive Variables  alpha, psi, rho, phi;

* Set initial values of alpha and rho
alpha.l = 0.001;
rho.l = 1;

* If we're forced to commit to the test, the optimal strategy when we have no choice is to commit all of the samples to the task and hope for a rejection.
* n.lo = 1e-3;

* maximum sample size needs to correspond to data generating function
* n.up = 10;

* n.fx = 1;
* phi.fx = 0.00475;

alpha.lo = 1e-6;
* rho.fx = 1;

Equations
    obj      "objective function"
    power    "power constraint"
    ero_c    "ero constraint"
    mfdr1    "mfdr control constraint 1"
    mfdr2    "mfdr control constraint 2"
    cost     "total experiment cost"
    ante     "player is indifferent as to whether to conduct test"
    alpha_W "allocated ante must be less than current wealth";


obj..      er1     =e= (q0*alpha + (1-q0)*rho) * psi ;

mfdr1..    psi     =l= phi/rho + mfdr_alpha ;

mfdr2..    psi     =l= phi/alpha + mfdr_alpha - 1;

ero_c..    phi/rho =e= phi/alpha - 1;

power..    rho     =e= 1 - normcdf( ((mu0-muA)*sqrt(n)/sigma) + normicdf(1-alpha,0,1), 0, 1);

alpha_W..  phi     =l= 0.1*Wa;

cost..     Wd      =g= n*c;


* The "player" is indifferent as to whether to conduct the test or not given a belief about the prior probability of the hypothesis (belief about whether nature has placed theta in H0 or HA).
ante..     q0*(alpha*(-phi+psi) + (1-alpha)*(-phi)) + (1-q0)*(rho*(-phi+psi) + (1-rho)*(-phi)) =e= 0;


Model ero / obj mfdr1 mfdr2 power ero_c cost ante alpha_W /;

solve ero using nlp maximizing er1;

display alpha.l, psi.l, rho.l, n.l, phi.l;
