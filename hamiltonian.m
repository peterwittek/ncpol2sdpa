%Hamiltonian - The Hamiltonian described in [1] solved with SeDuMi
%
%[1] Corboz, P.; Evenbly, G.; Verstraete, F. & Vidal, G. (2009), 
%    Simulation of interacting fermions with entanglement renormalization.
%    arXiv:0904.4151
%
% Other m-files required: Yalmip, SeDuMi, ncmoments.m, momentstosdpa.m
%                         getneighbors.m lineartolattice.m
% MAT-files required: none
%
% Author: Peter Wittek
% April 2013; Last revision: 11-Apr-2013


%------------- BEGIN CODE --------------

% Explicit definition that i will be sqrt(-1) in the code below
i=sqrt(-1);

% The dimension of the lattice in each direction
ld=2;

% Define non-commutative variables.
% Twice the number is required, as imaginary components
% will be handled manually. An operator C will be represented as
% C=X(r)+i*X(r+1)
n = 2*ld*ld;
X = ncvar(n,1);

% Parameters for the Hamiltonian
gamma=1;lambda=2;

obj=0; equalities = [];
% Constructing the Hamiltonian and the constraints
for r=1:2:n
    % Neighbors of r in the lattice
    neighbors=getneighbors(r, ld);
    for k=1:size(neighbors)
        s=neighbors(k);
        % The Hamiltonian
        obj = obj + ...
          (X(r)-i*X(r+1))*(X(s)+i*X(s+1))+(X(s)-i*X(s+1))*(X(r)+i*X(r+1)) + ...
          gamma*((X(r)-i*X(r+1))*(X(s)-i*X(s+1))+(X(s)+i*X(s+1))*(X(r)+i*X(r+1))) + ...
          2*lambda*((X(r)-i*X(r+1))*(X(r)+i*X(r+1)));
        % Equality constraints
        equalities = [ equalities; ...
          (X(r)+i*X(r+1))*(X(s)+i*X(s+1))+(X(s)+i*X(s+1))*(X(r)+i*X(r+1));...
          (X(r)+i*X(r+1))*(X(s)-i*X(s+1))+(X(s)-i*X(s+1))*(X(r)+i*X(r+1))];

    end
end

% There are no inequality constraints
inequalities = [];

%Order of relaxation (must be at least 2)
order = 2;

[Fmoments, M] = ncmoments(X, inequalities, equalities, order);

yalmip_options = sdpsettings('solver','sedumi','removeequalities ', 1,'relax',1,'saveduals',0);
sol=solvesdp(Fmoments,obj,yalmip_options);

% Obtaining the moment matrix
Mrel = relaxdouble(M{order+1});
