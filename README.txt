Compiling and linking: 2fortran

Running: te <config-file>
		Ex. te ../cfg/config.csv

Format config file:

First line: Number of time steps
Next lines: # as first letter ==> Comment line, ignored
<time-step>	<fault-number> <ON/OFF-1/0>

Ex.

385600
  # comment 0
100			3           	1
  # comment 1
200	5	1
# comment 2
 # comment 3
400	3	0
500	5	1

