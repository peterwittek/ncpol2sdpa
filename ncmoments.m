function [Fmoments]=ncmoments(X, inequalities, equalities, ord)
%NCMOMENTS - Converts noncommutative variables and sets of inequalities 
% and equalities to moments and localizing matrices
%
%Based on the noncommutative example in Bermuja:
%http://math.berkeley.edu/~philipp/Software/Moments
%
% Syntax:  [Fmoments] = ncmoments(X, inequalities, equalities, ord)
%
% Inputs:
%    X            - Noncommutative variables (ncvar type from Yalmip)
%    inequalities - Inequalities described in terms of X
%    equalities   - Equalities described in terms of X
%    ord          - Order of SDP relaxation
%
% Outputs:
%    Fmoments - Moments and localizing matrices
%
% Other m-files required: none
% MAT-files required: none
%

% Author: Philipp Rostalski, Peter Wittek
% December 2012; Last revision: 13-Dec-2012

%------------- BEGIN CODE --------------

% Noncommutative monomial vectors needed to set up moment matrices
mon = monolist(X,ord);

% Define the highest order moment matrix (symmetric part)
Maux = 1/2*(mon*mon'+(mon*mon').'); % I

% Construct all submatrices of Maux, s.t. 'M_s(y)' is stored in M{s+1}
for i=0:ord
    n_i(i+1) = 2^(i+1)-1;
    M{i+1} = Maux(1:n_i(i+1),1:n_i(i+1)); % compute moment matrices of lower order
end
% Add semidefinite constraints for the moment matrix
% (a nonsymetric moment matrix is PSD iff its symmetric part is).
F = set(M{ord+1} > 0); 

for j=1:length(inequalities)
    % Define the localization matrix for the inequality constraint
    % XXX gM = g*M{end-1};
    gM=mon(1:n_i(i))*inequalities(j)*mon(1:n_i(i))';

    % Add semidefinite constraints for the localizer
    F = F + set(gM > 0);
end

%% Computing a "Sylvester"-type matrix consisting of all polynomials and its shifts 
% of degree deg(pol*x^alpha) <= 2t

vec=[]; % initialize set of all linear equations and its shifted
        % form (multiplied by some monomials) in a vector

Feqconstr = set([]); % initialize constraint set (for SOLVESDP.m)

% Create all element-wise constraints, and put its shift in a vector 
for i = 1:length(equalities)
     % add all possible equality constraints of degree smaller or equal
     % to t
     % XXXX Localizer = pol(i)*monolist(X,2*t-degree(pol(i)));     
     Localizer = monolist(X,ord-ceil(1/2*degree(equalities(i))))*equalities(i)*monolist(X,ord-ceil(1/2*degree(equalities(i))))';
     
     vec = [vec;Localizer];
     Feqconstr = Feqconstr + set(Localizer==0); % and add as constraints
end


% Collect all constraints
Fmoments = F + Feqconstr;

%------------- END OF CODE --------------
