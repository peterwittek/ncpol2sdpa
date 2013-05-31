function momentstosdpa(filename, Fmoments, obj)
%MOMENTSTOSDPA - Converts SeDuMi-type moments to SDPA sparse input files
%
% Syntax:  momentstosdpa(filename, Fmoments, obj)
%
% Inputs:
%    filename - The name of the SDPA sparse file; must end with ".dat-s"
%    Fmoments - Moments and localizing matrices in SeDuMi format
%    obj      - Objective function
%
% Other m-files required: Yalmip, SeDuMi
% MAT-files required: none
%

% Author: Peter Wittek
% December 2012; Last revision: 13-Dec-2012

%------------- BEGIN CODE --------------

yalmip_options = sdpsettings('solver','sedumi','removeequalities ', 1,'relax',1);
yalmip_options.pureexport = 1;
[interfacedata] = solvesdp(Fmoments,obj,yalmip_options); 

F_struc = interfacedata.F_struc;
K       = interfacedata.K;
c       = interfacedata.c;
ub      = interfacedata.ub;
lb      = interfacedata.lb;

% Bounded variables converted to constraints
if ~isempty(ub)
    [F_struc,K] = addbounds(F_struc,K,ub,lb);
end

% Convert from internal (sedumi) format
[mDIM, nBLOCK, bLOCKsTRUCT, c, F] = sedumi2sdpa(F_struc,c,K);

%% Write file
datei = fopen(filename, 'w');

%Write header
var = [num2str(size(F, 2)-1) ' =mDIM'];
    %length(bLOCKsTRUCT) bLOCKsTRUCT ]);
%var = regexprep(var, ' *', ' ');
fprintf(datei, '%s\n', var);
var = [num2str(length(bLOCKsTRUCT)) ' =nBlock'];
fprintf(datei, '%s\n', var);
var = [num2str(bLOCKsTRUCT) ' =bLOCKsTRUCT'];
fprintf(datei, '%s\n', var);

%Write c
var = num2str(c');
var = regexprep(var, ' *', ', ');
fprintf(datei, '%s\n', var);

%Write elements
for k=1:size(F, 2)
    for block=1:size(F, 1)
        [i,j,v]=find(F{block,k});
        if bLOCKsTRUCT(block)<0
            j=i;
        end
        for x=1:length(v)
            var = num2str([ k-1 block i(x) j(x) v(x)]);
            var = regexprep(var, ' *', ' ');
            fprintf(datei, '%s\n', var);
        end
    end
    if block ~= size(F, 1) % prevent a empty line at EOF
        % OUTPUT newline
        fprintf(datei, '\n');
    end
end
% Closing file
fclose(datei);

%------------- END OF CODE --------------
