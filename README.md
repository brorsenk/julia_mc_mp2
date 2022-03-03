# julia_mc_mp2
Multicomponent MP2 using the Julia programming language

Requires the PySCF fork of the Yang group that implements multicomponent HF.
We have also contributed a multicomponent MP2 method to the PySCF for that 
is used by this Julia code to compare timings.

The fork be found here: https://github.com/theorychemyang

Only does the electron-proton integral transformation and energy calculations 
using Julia. Multicomponent HF is performed using PySCF.
