## Description
-`con2vert()` currently does not check to see if the initial interior point, which is required for the algorithm to work, is actually in the region, which will cause errors in some cases.

-Also, some combinations of stoichiometric matrix and feed vector seem to fail with a divide by zero error. Need to figure out what is the cause of this behaviour and fix it. (Unbounded regions perhaps?)
