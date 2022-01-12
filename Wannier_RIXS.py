import numpy as np

import math

import matplotlib.pyplot as plt

import numba as nb

import datetime

from matplotlib.colors import LinearSegmentedColormap


# 32 orbitals in a cell, 4X4X4 cell in real space

# index: 1-32 (cell1), then 33-64 cell 2 in (x,y,z) = (1,1,2), 

# 65-96 cell 3 in (x,y,z) = (1,1,3)

# index: the same repeat for (x,y,z) = (1,2,1) and so forth



startEint, endEint = -10, 15

startEloss, endEloss, step = 0, 25, 200

nbands = 34

nele_per_unit_cell = 18 # 18 spinless electron in one unit cell

a = 0.1 # lifetime (delta function)

b = 0.25 # intermediate lifetime

Uc = 2.5 # core hole potential

seedname = "Li2CO3"


nele = nele_per_unit_cell*64 

ncstart = nele

ncend = nbands*64

Ex, Ey = np.mgrid[startEint: endEint: (step)*1j,

                    startEloss: endEloss: (step)*1j]

Ey_s = np.mgrid[startEloss: endEloss: (step)*1j]

Ex_s = np.mgrid[startEint: endEint: (step)*1j]



def RIXS_Lithium_Carbonate(filehandle):



    # ind run from 1 to 8 to cover all c labels

    # ind = 1

    # mydir = '/global/cscratch1/sd/chunjing/projects/Li2CO3-ED/Li2CO3_v1.3_pol/'

    # seedname = 'Li2CO3'



     # number of bands in one unit cell

    

    #ncstart = nele + (ind - 1)*2*64

    #ncend = nele + ind*2*64



    

    """

    Matrix elements

    Previously rixs_core. Because it can be split into two: rixs_c_in and

    rixs_v.

    """

    t0 = datetime.datetime.now()

    rixs_c_i = np.zeros((3, ncend-ncstart, nbands*64-nele), dtype = np.complex128)

    rixs_elastic_c_i = np.zeros((3, 3, ncend-ncstart, nbands*64-nele), dtype = np.complex128)

    rixs_i_in = np.zeros((nbands*64-nele, step), dtype = np.complex128)

    rixs_c_in = np.zeros((3, ncend-ncstart, step), dtype = np.complex128)

    rixs_v = np.zeros((3, nele), dtype = np.complex128)

    # Matrix element for elastic line

    rixs_elastic = np.zeros((3,3,ncend-ncstart,step), dtype=np.complex128) 

    # Final result

    rixs_int = np.zeros((step, step), dtype=np.float64)

    print("Initial dimension: ")

    print("rixs_c_i: "+str(rixs_c_i.shape))

    print("rixs_elastic_c_i: "+str(rixs_elastic_c_i.shape))

    print("rixs_i_in: "+str(rixs_i_in.shape))

    print("rixs_c_in: "+str(rixs_c_in.shape))

    print("rixs_v: "+str(rixs_v.shape))

    print("rixs_elastic: "+str(rixs_elastic.shape))



    # ground state and final state Hamiltonian w/o core-hole

    ham = np.zeros((nbands*4*4*4, nbands*4*4*4), dtype=np.complex128)



    # intermediate state Hamiltonian with core-hole at site 0

    hamp = np.zeros((nbands*4*4*4, nbands*4*4*4), dtype=np.complex128)



    dat = np.zeros((nbands,nbands), dtype = np.complex128)

    #f = open(mydir + seedname + '_hr.dat', "r")

    f = filehandle

    f.readline()

    f.readline()

    line = f.readline()

    nrep = int(line.split()[0])

    for i in range(int((int(line.split()[0])-1)/15)+1):

        f.readline()

    for i in range(nrep):

        print(i, nrep)

        dat = np.zeros((nbands,nbands), dtype = np.complex128)

        line = f.readline()

        nx, ny, nz, oa, ob, val, val_im = line.split()

        nx, ny, nz = int(nx), int(ny), int(nz)

        dat[int(oa)-1,int(ob)-1] = float(val)+float(val_im)*1j

        for j in range(nbands*nbands-1):

            line = f.readline()

            oa, ob, val, val_im = line.split()[3:]

            dat[int(oa)-1,int(ob)-1] = float(val)+float(val_im)*1j

        for jx in range(4):

            for jy in range(4):

                for jz in range(4):

                    j = jx*16+jy*4+jz

                    k = np.mod(jx+nx,4)*16+np.mod(jy+ny,4)*4+np.mod(jz+nz,4)

                    ham[j*nbands: (j+1)*nbands, k*nbands: (k+1)*nbands] = dat

                    hamp[j*nbands: (j+1)*nbands, k*nbands: (k+1)*nbands] = dat





    # core hole at 1st unit cell. Some pz, px, py of an oxygen atom.

    hamp[0, 0] -= Uc

    hamp[1, 1] -= Uc

    hamp[2, 2] -= Uc 

    # maybe eigh?

    e0s, v0s = np.linalg.eigh(ham)

    print(e0s[nele - 3: nele + 3])

    eis, vis = np.linalg.eigh(hamp)

    print(eis[nele - 3: nele + 3])



    E0 = sum(e0s[:nele]) # Ground state energy

    mat0 = np.zeros((nele+1, nele+1), dtype=np.complex128) # A_c in paper



    Xi = np.dot(np.conjugate(np.transpose(v0s)), vis)

    #ZGEMM: BLAS matrix matrix multiply

    Xi = np.transpose(Xi)

    mat0[:nele, :nele] = Xi[:nele, :nele]

    E_intermediate_min = sum(eis[:nele])

    t1 = datetime.datetime.now()

    print("Preprocessing time cost: " + str(t1-t0))

    """

    Calculate RIXS intermediate matrix element.

    We want to calculate rixs_c_in and rixs_v, rixs_c_in = rixs_c_i.dot(rixs_i_in)

    For given final state (v,c), the matrix element is rixs_c_in[poli, c, :]*rixs_v[polf, v].

    """

    print("Preparing intermediate states...")

    get_rixs_c_i(rixs_c_i, rixs_elastic_c_i, nbands, ncstart, ncend, nele, v0s, mat0, Xi)

    t2 = datetime.datetime.now()

    print("(core, intermediate) done! This cost: " + str(t2-t1))

    get_rixs_i_in(rixs_i_in, E_intermediate_min, eis, E0)

    t3 = datetime.datetime.now()

    print("(intermediate, E_out) done! This cost: " + str(t3-t2))

    rixs_c_in = rixs_c_i.dot(rixs_i_in)



    rixs_elastic = rixs_elastic_c_i.dot(rixs_i_in)

    rixs_v = v0s[0:3, 0:nele]

    t4 = datetime.datetime.now()

    print("Intermediate state matrix elements done! In total this costs: " + str(t4-t1))

    print("Dimensions: ")

    print("rixs_c_i: "+str(rixs_c_i.shape))

    print("rixs_elastic_c_i: "+str(rixs_elastic_c_i.shape))

    print("rixs_i_in: "+str(rixs_i_in.shape))

    print("rixs_c_in: "+str(rixs_c_in.shape))

    print("rixs_v: "+str(rixs_v.shape))

    print("rixs_elastic: "+str(rixs_elastic.shape))

    print("Calculating spectra...")

    update_spectra(rixs_int, rixs_c_in, rixs_v, rixs_elastic, e0s)

    t5 = datetime.datetime.now()

    print("Updating spectra costs: " + str(t5-t4))

                    

    # np.savetxt(mydir + "RIXS_Ex_O1_nc" + str(ind) + ".csv", Ex_s, delimiter=" ")

    # np.savetxt(mydir + "RIXS_Ey_O1_nc" + str(ind) + ".csv", Ey_s, delimiter=" ")

    return rixs_int



@nb.jit(nopython=True)

def get_rixs_c_i(rixs_c_i, rixs_elastic_c_i, nbands, ncstart, ncend, nele, v0s, mat0, Xi):

    """

    Using the trick of rank 1 update when calculating the determinant

    """

    u = np.zeros(nele+1, dtype = np.complex128)

    u[-1] = 1.0

    for nc in range(ncstart, ncend):

        nc_left_index = nc - ncstart

        # Originally, put mat0 close to A_c^c. This matrix should be nearly unitary.

        mat0[:nele, nele] = Xi[:nele, nc]

        matele_i = np.conj(v0s[0:3, nc])

        mat0[nele, :nele] = Xi[nc, :nele]

        mat0[nele, nele] = Xi[nc, nc]

        Ac_0 = np.linalg.det(mat0)

        Ac_0_inv_last_col = np.linalg.solve(mat0, u)

        ac_0 = np.abs(Ac_0)

        #print ac_0

        # if ac_0 < 0.1:

        #     print "Warning: ac_0 too small."

        #     print nc

        #     print ac_0

        for i in range(nele, nbands*64):

            i_left_index = i - nele

            #mat0[nele, :nele] = Xi[i, :nele]

            #mat0[nele, nele] = Xi[i, nc]

            #Ac = np.linalg.det(mat0)

            Ac = (1 + np.dot(Xi[i, :nele]-Xi[nc, :nele], Ac_0_inv_last_col[:-1]) + (Xi[i, nc]-Xi[nc, nc])*Ac_0_inv_last_col[-1])*Ac_0

            # Ein = E_intermediate_min + eis[i] - E0

            for poli in range(3):

                rixs_c_i[poli, nc_left_index, i_left_index] = np.abs(Ac)**2*matele_i[poli]

                for polf in range(3):

                    rixs_elastic_c_i[poli, polf, nc_left_index, i_left_index] = np.abs(Ac)**2*matele_i[poli]*np.conj(matele_i[polf])

@nb.jit(nopython=True)

def get_rixs_i_in(rixs_i_in, E_intermediate_min, eis, E0):

    for i in range(nele, nbands*64):

        i_left_index = i - nele

        E_intermediate = E_intermediate_min + eis[i] - E0

        rixs_i_in[i_left_index,:] = 1.0/(Ex_s - E_intermediate + b*1j)







        

@nb.jit(nopython=True)

def update_spectra(rixs_int, rixs_c_in, rixs_v, rixs_elastic, e0s):

    max_width = 10*np.sqrt(a)

    # Auxiliary variables

    rixs_in = np.zeros((step, (ncend - ncstart)*nele), dtype = np.float64)

    rixs_loss = np.zeros(((ncend - ncstart)*nele, step), dtype = np.float64)

    for nc in range(ncstart, ncend):

        nc_index = nc - ncstart

        for nv in range(nele):

            Eloss = e0s[nc] - e0s[nv]

            # Update rixs_loss

            for el in range(step):

                rixs_loss[nc_index*nele+nv, el] += 1.0/(a+(Ey_s[el] - Eloss)**2/a)

            # Update rixs_in

            for ei in range(step):

                for polf in range(3):

                    for poli in range(3):

                        polfac = 1.5 # What's this?

                        if poli == polf:

                            polfac = 1.0

                        rixs_in[ei, nc_index*nele+nv] += polfac * np.absolute(rixs_c_in[poli, nc_index, ei]*rixs_v[polf, nv]) **2

                        # rixs_int[ei,el] += polfac * np.absolute(rixs_c_in[poli, nc_index, ei]*rixs_v[polf, nv]) **2 / \

                        #                 ((Ey_s[el] - Eloss)**2/a + a)

    rixs_int += np.dot(rixs_in, rixs_loss)

    print("Inelastic contributions done!")

    Eloss = 0.0

    for nc in range(ncstart, ncend):

        nc_index = nc - ncstart

        for el in range(step):

            if np.abs(Ey_s[el] - Eloss) > max_width:

                continue

            for ei in range(step):

                for polf in range(3):

                    for poli in range(3):

                        polfac = 1.5

                        if poli == polf:

                            polfac = 1.0

                        rixs_int[ei, el] += polfac * np.absolute(rixs_elastic[poli, polf, nc_index, ei]) **2 / \
                                            ((Ey_s[el] - Eloss)**2/a + a)

    print("Elastic contributions done!")



    #plt.plot(specX, specY)

    #plt.show()



"""

This part is deprecated.



@nb.jit(nopython=True)

def compute_i_intermediate(i, nele, step, mat0, Xi, ncstart, ncend, endEloss, Ein, e0s, v0s, Ex_s, rixs_core, rixs_elastic, b):

    for nc in range(ncstart, ncend): # c label

        #for nc in range(nele, nbands * 64): 

        # nc label should run this range if not doing manually parallelization

        mat0[:nele, nele] = Xi[:nele, nc]

        mat0[nele, nele] = Xi[i, nc]

        Ac = np.linalg.det(mat0)

        ac = np.absolute(Ac)

        #print "At i = %d & nc = %d, Ac is %f.\n" % (i, nc, ac)

        

        #0 represent the first orbital in the wannier hamiltonian

        matele_i = np.conj(v0s[0:3, nc]) #+ v0s[1, nc] + v0s[2, nc]



        # intermediate state | final state dipole matrix element

        for nv in range(nele): # valence band hole

            Eloss = e0s[nc] - e0s[nv]

            if Eloss > endEloss:

                continue



            # the first index 0, 1, 2 represents pz, px and py orbitals

            matele_f = v0s[0:3, nv] #+ v0s[1, nv] + v0s[2, nv] 

            prev = np.outer(matele_i, matele_f)*np.conj(Ac)*Ac

            for poli in range(3):

                for polf in range(3):

                    rixs_core[poli, polf, nc-ncstart, nv, :] += 1.0*prev[poli,polf]/(Ex_s - Ein + b*1j)

                #    rixs_core[poli,polf,:] += 1.0*prev[poli,polf]/(Ex_s - Ein + b*1j)

        prev = np.outer(np.conj(matele_i), matele_i)*np.conj(Ac)*Ac

        for poli in range(3):

            for polf in range(3):

                rixs_elastic[poli, polf, :] += 1.0*prev[poli,polf]/(Ex_s - Ein + b*1j)

"""

def plotting(datafile, plot_filename):

    # combine them and build a new colormap
    colors1 = plt.cm.terrain(np.linspace(0., 0.75, 128))
    colors2 = plt.cm.terrain(np.linspace(0.75, 1, 128))
    colors = np.vstack((colors1, colors2))
    mymap = LinearSegmentedColormap.from_list('my_colormap', colors)    
    
    rixs_int = datafile
    rixs_int_p = np.zeros((200,400))
    for i in range(200):
        rixs_int_p[i, i:(i+200)] = rixs_int[i,::-1]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111) 
    im = plt.imshow(rixs_int_p[:,100:100+200], interpolation='bilinear', \
                extent=[-20+538.5,5+538.5, -15+538.5,10+538.5],
                origin='lower',vmax=rixs_int.max()/1.1, vmin=rixs_int.min(), \
                cmap = mymap)
    plt.colorbar(orientation ='vertical')
    plt.xlabel('Emission Energy (eV)')
    plt.ylabel('Incident Energy (eV)')
    plt.savefig(plot_filename)
    plt.show()

def main():

    filehandle = open(seedname + "_hr.dat")

    result = RIXS_Lithium_Carbonate(filehandle)

    filehandle.close()

    #plt.imshow(result, colorbar = 'terrain')
    #plt.show()
    ##plt.savefig("RIXS_Li2CO3_O_K-edge.png")
    #plt.colorbar()

    np.savetxt(seedname + "_O_K-edge_RIXS" + ".csv", result, delimiter=" ")

    plotting(result, seedname + "_O_K-edge_RIXS.png")

if __name__== "__main__":

    main()





