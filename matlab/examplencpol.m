%EXAMPLE - An example solved with SeDuMi or exported to
% sparse SDPA format for scalable computation. The description 
% of the example is in the following paper:
%
% Pironio, S.; Navascués, M. & Acín, A. Convergent relaxations of 
% polynomial optimization problems with noncommuting variables SIAM Journal
% on Optimization, SIAM, 2010, 20, 2157-2180

% Other m-files required: Yalmip, SeDuMi, ncmoments.m, momentstosdpa.m
% MAT-files required: none
%

% Author: Peter Wittek
% December 2012; Last revision: 31-May-2012

%------------- BEGIN CODE --------------

% Define non-commutative variables
n = 2;
X = ncvar(n,1);

% Cost function
obj = X(1)*X(2)+X(2)*X(1);

% Equality constraints
equalities = X(1)^2-X(1);
% Commutative variant
%equalities = [ X(1)^2-X(1); X(2)*X(1)-X(1)*X(2) ];


% Inequality constraints
inequalities = -X(2)^2+X(2)+1/2;

%Order of relaxation
order = 2;

[Fmoments] = ncmoments(X, inequalities, equalities, order);

%There are two options:

%1) The problem is of small or moderate size. Solve the problem in Matlab.
%   Note that noncommutative variables are experimental, and evaluating the
%   the objective function by double(obj) will not return the correct result.    
yalmip_options = sdpsettings('solver','sedumi','removeequalities ', 1,'relax',1,'saveduals',0);
solvesdp(Fmoments,obj,yalmip_options);

%2) The problem is large. Export it to SDPA sparse format and solve the
%   problem with SDPARA.
momentstosdpa('example.dat-s', Fmoments, obj);

%------------- END OF CODE --------------
