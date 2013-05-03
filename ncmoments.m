function [Fmoments, M]=ncmoments(X, inequalities, equalities, order)
%NCMOMENTS - Converts noncommutative variables and sets of inequalities 
% and equalities to moments and localizing matrices
%
%Based on the noncommutative example in Bermuja:
%http://math.berkeley.edu/~philipp/Software/Moments
%
% Syntax:  [Fmoments] = ncmoments(X, inequalities, equalities, order)
%
% Inputs:
%    X            - Noncommutative variables (ncvar type from Yalmip)
%    inequalities - Inequalities described in terms of X
%    equalities   - Equalities described in terms of X
%    order        - Order of SDP relaxation
%
% Outputs:
%    Fmoments - Moments and localizing matrices
%
% Other m-files required: none
% MAT-files required: none
%

% Author: Philipp Rostalski, Peter Wittek
% December 2012; Last revision: 03-May-2013

%------------- BEGIN CODE --------------

% Noncommutative monomial vectors needed to set up moment matrices
mon = monolist(X,order);
    
tempMon=mon*mon';

% Define the highest order moment matrix (symmetric part)
Maux = 1/2*(tempMon+(tempMon).'); % I

% Construct all submatrices of Maux, s.t. 'M_s(y)' is stored in M{s+1}
for i=0:order
    n_i(i+1) = 2^(i+1)-1;
    M{i+1} = Maux(1:n_i(i+1),1:n_i(i+1)); % compute moment matrices of lower order
end
% Add semidefinite constraints for the moment matrix
% (a nonsymetric moment matrix is PSD iff its symmetric part is).
F = set(M{order+1} > 0); 

% ...but we also want M to be a symmetric matrix.
F = F+ set(tempMon==(tempMon).');

for j=1:length(inequalities)
    % Define the localization matrix for the inequality constraint
    % XXX gM = g*M{end-1};
    gM=mon(1:n_i(i))*inequalities(j)*mon(1:n_i(i))';

    % Add semidefinite constraints for the localizer
    F = F + set(gM > 0);
end

%% Computing a "Sylvester"-type matrix consisting of all polynomials and its shifts 
% of degree deg(pol*x^alpha) <= 2t

Feqconstr = set([]); % initialize constraint set (for SOLVESDP.m)

% Create all element-wise constraints, and put its shift in a vector 
for i = 1:length(equalities)
     % add all possible equality constraints of degree smaller or equal
     % to t
     index=n_i(order-ceil(1/2*degree(equalities(i)))+1);
     Localizer = mon(1:index)*equalities(i)*mon(1:index)';
     
     Feqconstr = Feqconstr + set(Localizer==0); % and add as constraints
end


% Collect all constraints
Fmoments = F + Feqconstr;

%------------- END OF CODE --------------
