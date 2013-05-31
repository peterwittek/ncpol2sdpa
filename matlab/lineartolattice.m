function [coords] = lineartolattice(index, dimension)
%LINEARTOLATTICE - Translates a linear index to 2D coordinates in a lattice
%
% Syntax:  [coords] = lineartolattice(index, dimension)
%
% Inputs:
%    index        - Linear index in an array of the location
%    dimension    - Horizontal dimension of the lattice
%
% Outputs:
%    [coords]     - The coordinates in the 2D lattice
%
% Other m-files required: none
% MAT-files required: none
%

% Author: Peter Wittek
% April 2013; Last revision: 11-Apr-2013

%------------- BEGIN CODE --------------

  coords(1)=mod(index,dimension);
  coords(2)=ceil(index/dimension);
  if coords(1)==0
      coords(1)=dimension;
  end
  