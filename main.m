%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Downs, J.J. and Vogel, E.F., 1993. A plant-wide industrial process control problem.
% Computers & chemical engineering, 17(3), pp.245-255.
%
% Online: http://www.abo.fi/~khaggblo/RS/Downs.pdf
%
% @article{downs1993plant,
%  title={A plant-wide industrial process control problem},
%  author={Downs, James J and Vogel, Ernest F},
%  journal={Computers \& chemical engineering},
%  volume={17},
%  number={3},
%  pages={245--255},
%  year={1993},
%  publisher={Elsevier}
% }
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compile: mex tesub.F
% Call from Matlab: [X, Y] = tesub(RANDOMSEED, NPTS, EVENTS);
%


RANDOMSEED = 4651207995D0;	% Seed of Fortran random number generator
NPTS = 2160000;			% Number of simulation steps (seconds), 180 sec = 1 sample

% EVENTS (fault on/off)
% # 180 = 1 sample
% # Timestamp     Fault                   State(1=switch on/0=switch off)
% # =====================================================================
EVENTS = [
180		1			1;
216180		1			0;
216180		2			1;
432180		2			0;
432180		3			1;
648180		3			0;
648180		4			1;
864180		4			0;
864180		5			1;
1080180		5			0;
1080180		6			1;
1296180		6			0;
1296180		7			1;
1512180		7			0;
1512180		8			1;
1728180		8			0;
1728180		9			1;
1944180		9			0;
1944180		10			1;
];

if 0
NPTS = 3600;
EVENTS = [
180		1			1;
900		1			0;
900		2			1;
1800		2			0;
1800		4			1;
];
end

if 0
NPTS = 432180;
EVENTS = [
180		1			1;
216180		1			0;
216180		2			1;
432180		2			0;
432180		3			1;
];
end


if 1	% ORIGINAL CONFIGURATION OF temain_mod.f.txt
NPTS = 172800;
EVENTS = [
28800		12			1;
];
end

% 

% Output
% Generate data matrix X and label matrix Y
% Lines = samples
% Columns of X = features (see TE simulator)
%    First 12 features = Manipulated variables XMV(1) --- XMV(12) (Table 3. of paper)
%    Features 13-53 = Measured variables XMEAS(1) --- XMV(22) (Table 4. of paper)
%
% Columns of Y = 20 class labels (see TE simulator)

[X, Y] = tesub(RANDOMSEED, NPTS, EVENTS);

%csvwrite('X.csv',X); csvwrite('Y.csv',Y);

