$funclibin stolib stodclib
Functions normcdf     / stolib.cdfNormal /
          normicdf    / stolib.icdfNormal / ;

option nlp=conopt;

Scalar
    muA1        "alternative mean1"      / 2.0 /
    muA2        "alternative mean2"      / 2.0 /
    mu0        "null mean"             / 0.0 /
    sigma1     "std dev of data"       / 1.0 /
    sigma2     "std dev of data"       / 1.0 /
    q1         "prior of the first null"     / 0.1 /
    q2         "prior of the second null"     / 0.1 /
    mfdr_alpha "mFDR control level"    / 0.05 /
    Wd         "dollar wealth"         / 100000000.0 /
    Wa         "alpha wealth"          / 0.0475 /
    c          "cost per sample"       / 1.0 /;
    
    
Variables
    alpha1 "level of test 1"
    alpha2 "level of test 2"
    psi1   "reward for successful rejection 1"
    psi2   "reward for successful rejection 2"  
    n1     "number of samples 1"
    n2     "number of samples 2"
    phi1   "cost of test 1"
    phi2   "cost of test 2"
    Wa2    "update alpha" 
    TT1    "Test Test1"
    TT2    "Test Test2"	
    TT3    "Test Test3"	
    TT4    "Test Test4"
    TN1    "Test Not Test1"	
    TN2    "Test Not Test2"
    NT1    "Not Test Test1"	
    NT2    "Not Test Test2";
	
Variables
    rho1     "power of the test 1"
    rho2     "power of the test 2"
    er1     "expected reward for test";
    
Positive Variables  alpha1, alpha2, psi1, psi2, rho1, rho2, phi1, phi2;

* Set initial values of alpha and rho
alpha1.l = 0.001;
rho1.l = 1;
alpha2.l = 0.001;
rho2.l = 1;

* If we're forced to commit to the test, the optimal strategy when we have no choice is to commit all of the samples to the task and hope for a rejection.
* n.lo = 1;

* maximum sample size needs to correspond to data generating function
* n1.up = 10;
* n2.up = 10;

* n.fx = 1;
* phi.fx = 0.00475;

alpha1.lo = 1e-6;
alpha2.lo = 1e-6;
* rho.fx = 1;

Equations
    obj      "objective function"
    power1    "power1 constraint"
    power2    "power2 constraint"
    ero_c1    "ero1 constraint"
    ero_c2    "ero2 constraint"
    mfdr11    "mfdr1 control constraint 1"
    mfdr12    "mfdr2 control constraint 1"
    mfdr21    "mfdr1 control constraint 2"
    mfdr22    "mfdr2 control constraint 2"	
    cost      "total experiment cost"
    P1	      "Probability 1"
    P2	      "Probability 2"	
    P3	      "Probability 3"	
    P4	      "Probability 4"	
    P5	      "Probability 5"	
    P6	      "Probability 6"	
    P7	      "Probability 7"	
    P8	      "Probability 8"		
    ante1     "player is indifferent as to whether to conduct test"
    ante2     "player is indifferent as to whether to conduct test"
    ante3     "player is indifferent as to whether to conduct test"	
    alpha_W1  "allocated ante must be less than current wealth"
    alpha_W2  "allocated ante must be less than current wealth"
    w_step    "Update w in expectation";		


obj..      er1     =e= (q1*alpha1 + (1-q1)*rho1) * psi1 + (q2*alpha2 + (1-q2)*rho2) * psi2 ;

mfdr11..    psi1     =l= phi1/rho1 + mfdr_alpha ;

mfdr21..    psi1    =l= phi1/alpha1 + mfdr_alpha - 1;

ero_c1..    phi1/rho1 =e= phi1/alpha1 - 1;

power1..    rho1     =e= 1 - normcdf( ((mu0-muA1)*sqrt(n1)/sigma1) + normicdf(1-alpha1,0,1), 0, 1);

alpha_W1..  phi1     =l= Wa;

mfdr12..    psi2     =l= phi2/rho2 + mfdr_alpha ;

mfdr22..    psi2     =l= phi2/alpha2 + mfdr_alpha - 1;

ero_c2..    phi2/rho2 =e= phi2/alpha2 - 1;

power2..    rho2     =e= 1 - normcdf( ((mu0-muA2)*sqrt(n2)/sigma2) + normicdf(1-alpha2,0,1), 0, 1);

alpha_W2..  phi2     =l= Wa2;

w_step..    Wa2      =e= Wa - phi1 + (q1*alpha1 + (1 - q1)*rho1) * psi1;

cost..     Wd      =g= (n1 + n2)*c;

P5..       TN1    =e= q1*alpha1 + (1-q1)*rho1;

P6..       TN2    =e= q1*(1-alpha1) + (1-q1)*(1-rho1);

P7..       NT1    =e= q2*alpha2 + (1-q2)*rho2;

P8..       NT2    =e= q2*(1-alpha2) + (1-q2)*(1-rho2);

P1..       TT1    =e= TN1*NT1;

P2..       TT2    =e= TN1*NT2;

P3..       TT3    =e= TN2*NT1;

P4..       TT4    =e= TN2*NT2;

* The "player" is indifferent as to whether to conduct the test or not given a belief about the prior probability of the hypothesis (belief about whether nature has placed theta in H0 or HA).
ante1..    0 =e= TT1*(-phi1 - phi2 + psi1 + psi2) + TT2*(-phi1 - phi2 + psi1) + TT3*(-phi1 - phi2 + psi2) + TT4*(-phi1 - phi2);

ante2..    0 =e= TN1*(-phi1 + psi1) + TN2*(-phi1); 

ante3..    0 =e= NT1*(-phi2 + psi2) + NT2*(-phi2); 



Model ero / obj mfdr11 mfdr21 mfdr12 mfdr22 power1 power2 cost ante1 ante2 ante3 alpha_W1 alpha_W2 ero_c1 ero_c2 w_step P1 P2 P3 P4 P5 P6 P7 P8 /;

solve ero using nlp maximizing er1;

display alpha1.l, psi1.l, rho1.l, n1.l, phi1.l, alpha2.l, psi2.l, rho2.l, n2.l, phi2.l;
