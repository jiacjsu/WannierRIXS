# WannierRIXS

### general description
This python3 codes implements the resonant inelastic x-ray scattering calculation for the O K-edge based on a Wannier orbital basis 
and a Slater determinant representation of the wavefunction. The algorithm and an example of the calculation for Li2CO3 has been described in this paper:  
["A Wannier orbital based method for resonant inelastic x-ray scattering simulation", Chunjing Jia, J. Phys.: Conf. Ser. 1290 012014 (2019)](https://iopscience.iop.org/article/10.1088/1742-6596/1290/1/012014/meta)

### input and output
This code takes an input file "seedname_hr.dat", which is the output file of Wannier90 that consists of all tight-binding Hamiltonian parameters. 
You are expected to change other parameters in the code as well. An example for Li2CO3 are shown below:

startEint, endEint = -10, 15 

startEloss, endEloss, step = 0, 25, 200

nbands = 34 # needs to be consistent with the number of bands in the "_hr.dat" file

nele_per_unit_cell = 18 # 18 spinless electron in one unit cell

a = 0.1 #  final state lifetime 

b = 0.25 # intermediate state lifetime

Uc = 2.5 # core hole potential in unit of eV

seedname = "Li2CO3" # needs to be consistent with "_hr.dat" file

The output files include a RIXS datafile and a RIXS image.
