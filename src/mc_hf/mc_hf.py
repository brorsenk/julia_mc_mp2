#!/usr/bin/env python3
  
import numpy
from pyscf import neo
import time

def get_mp2_quantities():

  mol = neo.Mole()
  mol.build(atom='''H 0 0 0; F 0 0 1.13; F 0 0 -1.13''', basis='ccpv5z', quantum_nuc=[0], charge=-1)
  mf = neo.HF(mol)
  energy = mf.scf()

  emp2_ee, emp2_ep = neo.MP2(mf).kernel()

  print('emp2_ee = ',emp2_ee)
  print('emp2_ep = ',emp2_ep)
  print('total neo-mp2 = ',energy+emp2_ee+emp2_ep)

  eri_ep_ao_ints, mo_coeff = neo.ao2mo_neo.ep_setup(mf)

  eri_ep = neo.ao2mo_neo.ep_ovov(mf)

  e_nocc = mf.mf_elec.mo_coeff[:,mf.mf_elec.mo_occ>0].shape[1]
  e_tot  = mf.mf_elec.mo_coeff[0,:].shape[0]
  e_nvir = e_tot - e_nocc

  p_nocc = mf.mf_nuc[0].mo_coeff[:,mf.mf_nuc[0].mo_occ>0].shape[1]
  p_tot  = mf.mf_nuc[0].mo_coeff[0,:].shape[0]
  p_nvir = p_tot - p_nocc

  eia = mf.mf_elec.mo_energy[:e_nocc,None] - mf.mf_elec.mo_energy[None,e_nocc:]
  ejb = mf.mf_nuc[0].mo_energy[:p_nocc,None] - mf.mf_nuc[0].mo_energy[None,p_nocc:]

  eri_ep = eri_ep.reshape(e_nocc, e_nvir, p_nocc, p_nvir)

  e_nao = int(mf.mf_elec.mol.nao_nr())
  p_nao = int(mf.mf_nuc[0].mol.nao_nr())

  return eri_ep, eri_ep_ao_ints, eia, ejb, e_nocc, e_nvir, p_nocc, p_nvir, mo_coeff, e_nao, p_nao


