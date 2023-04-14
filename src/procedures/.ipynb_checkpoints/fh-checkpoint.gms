$funclibin stolib stodclib
Functions normcdf     / stolib.cdfNormal /
          normicdf    / stolib.icdfNormal / ;

option nlp=conopt;

Set
    j          "tests in batch"        /j1*j9/

Scalars
    mfdr_alpha "mFDR control level"    / 0.05 /
    Wd         "dollar wealth"         / 100 /
    Wa         "alpha wealth"          / 0.0475 /;

Parameters
    muA(j)     "alternative mean"      /set.j 2.0/
    mu0(j)     "null mean"             /set.j 0.0/
    sigma(j)   "std dev of data"       /set.j 1.0/
    q0(j)      "prior of the null"     / j1=0.9, j2=0.9, j3=0.9, j4=0.9, j5=0.9, j6=0.9, j7=0.9, j8=0.9, j9=0.9 /
    c(j)       "cost per sample"       /set.j 1.0/;
    
Variables
    alpha(j) "level of test"
    psi(j)   "reward for successful rejection" 
    n(j)     "number of samples"
    phi(j)   "cost of test"
    W(j)       "Wealth process";

Variables
    rho(j)     "power of the test"
    er1     "expected reward for test";
    
Positive Variables  alpha(j), psi(j), rho(j), phi(j);

* Set initial values of alpha and rho
alpha.l(j) = 0.04;
rho.l(j) = 1;
* phi.l(j) = .01;

* Set lower bounds to avoid numerical instability 
alpha.lo(j) = 1e-6;

* n.lo(j) = 1;


Equations
    obj      "objective function"
    power    "power constraint"
    ero_c(j) "ero constraint"
    mfdr1(j) "mfdr control constraint 1"
    mfdr2(j) "mfdr control constraint 2"
    cost     "constraint: total experiment cost"
    ante  "constraint: player is indifferent as to whether to conduct test"
    alpha_W  "constraint: allocated ante must be less than current wealth"
    alpha_w0 "initial wealth of first test"
    alpha_next "expected value of next test"
    pos_phi   "positive phi"
    pos_W     "positive wealth";


obj..      er1            =e= sum(j, (q0(j)*alpha(j) + (1-q0(j))*rho(j))*psi(j));
* obj..      er1            =e= sum(j, U(j)*alpha(j));

mfdr1(j).. psi(j)         =l= phi(j)/rho(j) + mfdr_alpha ;

mfdr2(j).. psi(j)         =l= phi(j)/alpha(j) + mfdr_alpha - 1;

ero_c(j).. phi(j)/rho(j)  =e= phi(j)/alpha(j) - 1;

power(j).. rho(j)         =e= 1 - normcdf( ((mu0(j)-muA(j))*sqrt(n(j))/sigma(j)) + normicdf(1-alpha(j),0,1), 0, 1);

alpha_W0..   W('j1')  =e= Wa;

alpha_W(j)..  phi(j)         =l= 0.1*W(j);

* pos_phi(j)..   phi(j)  =g= 0;

pos_W(j)..   W(j)  =g= 0;

alpha_next(j)$(not sameas(j,'j9')).. W(j+1)       =e= W(j) - phi(j) + (q0(j)*alpha(j) + (1-q0(j))*rho(j)) * psi(j);

cost..     Wd      =g= sum(j, n(j)*c(j));

* The "player" is indifferent as to whether to conduct the test or not given a belief about the prior probability of the hypothesis (belief about whether nature has placed theta in H0 or HA).
ante(j)..     q0(j)*(alpha(j)*(-phi(j)+psi(j)) + (1-alpha(j))*(-phi(j))) + (1-q0(j))*(rho(j)*(-phi(j)+psi(j)) + (1-rho(j))*(-phi(j))) =e= 0;

Model ero / obj mfdr1 mfdr2 power ero_c cost ante alpha_W alpha_W0 alpha_next pos_W/;
* Model ero / obj mfdr1 mfdr2 power ero_c cost alpha_W alpha_W0 alpha_next pos_phi pos_W/;
* Model ero / obj mfdr1 mfdr2 power cost alpha_W alpha_W0 alpha_next pos_phi pos_W ante/;

solve ero using nlp maximizing er1;

* display alpha.l(j), psi.l(j), rho.l(j), n.l(j), phi.l(j), W(j);