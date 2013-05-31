function neighbors = getneighbors(index, ld)
%GETNEIGHBORS - Gets the neighbors of a location given by a linear index in
% a 2D grid
%
% Syntax:  neighbors = getneighbors(r, ld)
%
% Inputs:
%    index        - Linear index in an array of the location
%    ld           - Horizontal dimension of the lattice
%
% Outputs:
%    neighbors    - The neighbors given in linear indices
%
% Other m-files required: lineartolattic.m
% MAT-files required: none
%

% Author: Peter Wittek
% April 2013; Last revision: 11-Apr-2013

%------------- BEGIN CODE --------------

    neighbors=[];
    % Note that a complex variable is actually two real variables
    coords=lineartolattice(index,2*ld);
    if (coords(1)>1)
        neighbors=[neighbors index-2];
    end
    if (coords(1)<2*ld-1)
        neighbors=[neighbors index+2];
    end
    if (coords(2)>1)
        neighbors=[neighbors index-2*ld];
    end
    % Vertically the row number is correct, no adjustment is necessary
    if (coords(2)<ld)
        neighbors=[neighbors index+2*ld];
    end