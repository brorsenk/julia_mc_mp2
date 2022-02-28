#!/usr/bin/env python3
  
#
#Copyright 2022 Kurt R. Brorsen
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
using PyCall
using BenchmarkTools
using Base.Threads
using LinearAlgebra
using Profile

BLAS.set_num_threads(1)

mm = pyimport("mc_hf.mc_hf")

eri_ep, eri_ep_ao_ints, eia, ejb, e_nocc, e_nvir, p_nocc, p_nvir, mo_coeff, e_nao, p_nao = mm.get_mp2_quantities()

const nao = e_nao + p_nao
const co_e = mo_coeff[ :, 1:e_nocc]
const cv_e = mo_coeff[ :, 1+e_nocc:e_nocc+e_nvir]
const co_n = mo_coeff[ :, 1+e_nocc+e_nvir:e_nocc+e_nvir+p_nocc]
const cv_n = mo_coeff[ :, 1+e_nocc+e_nvir+p_nocc:e_nocc+e_nvir+p_nocc+p_nvir]

function index_2(i,j)
  if(i>j)
    return div(i*(i-1),2)+j
  else
    return div(j*(j-1),2)+i
  end
end

function ep_energy(eri_ep, eia, ejb, p_nvir, p_nocc, e_nvir, e_nocc)
  emp2_ep = 0.0
  for a = 1:p_nvir, i = 1:p_nocc, b = 1:e_nvir, j= 1:e_nocc
           @inbounds emp2_ep += eri_ep[j,b,i,a]^2 / (eia[j,b] + ejb[i,a])
  end
  emp2_ep = 2.0 * emp2_ep
  return emp2_ep
end

function half_loop1!(tmp, i, j,  e_nao, p_nao, eri_ep_ao_ints, x1, y1, z1, mo3, mo4, indx3, indx4)
  ij = index_2(i,j) 
  kl=0
   for k = 1:p_nao
    for l = 1:k
      kl = index_2(e_nao+k,e_nao+l) 
      ijkl = index_2(ij,kl) 
      @inbounds x1[k,l] = eri_ep_ao_ints[ijkl]
      @inbounds x1[l,k] = x1[k,l]
    end
  end

  @inbounds mul!(y1, mo3,x1)
  @inbounds mul!(z1, y1, mo4)
  @inbounds tmp[:,ij] = vec(z1)

end

function half_loop2!(tei_mo, k, l, e_nao, mo1, mo2, indx1, indx2, indx3, indx4, x2,y2,z2, tmp)
  kl = (k-1)*indx4 + l
  ij = 0
   for i in 1:e_nao
     for j in 1:i
      ij = ij + 1
      @inbounds x2[i,j] = tmp[kl,ij]
      @inbounds x2[j,i] = x2[i,j]
    end
    end

  @inbounds mul!(y2, mo1, x2)
  @inbounds mul!(z2, y2, mo2)
  @inbounds tei_mo[:,:,k,l] = z2

end


function fast_conversion_general(nao, e_nao, p_nao, e_nocc, e_nvir, p_nocc, p_nvir, mo1, mo2, mo3, mo4, eri_ep_ao_ints )

  x1 = zeros(p_nao,p_nao,nthreads())
  y1 = zeros(p_nocc,p_nao,nthreads())
  z1 = zeros(p_nocc,p_nvir,nthreads())

  x2 = zeros(e_nao,e_nao,nthreads())
  y2 = zeros(e_nocc,e_nao,nthreads())
  z2 = zeros(e_nocc,e_nvir,nthreads())

  indx1 = size(mo1)[2]
  indx2 = size(mo2)[2]
  indx3 = size(mo3)[2]
  indx4 = size(mo4)[2]

  tei_mo = zeros(indx1, indx2, indx3, indx4)

  mo1_subset = transpose(mo1[1:e_nao,:])
  mo2_subset = mo2[1:e_nao,:]
  mo3_subset = transpose(mo3[1+e_nao:nao,:])
  mo4_subset = mo4[1+e_nao:nao,:]
 
  value1 = indx3*indx4
  value2 = div(nao*(nao+1),2)

  tmp = zeros(value1,value2)

  local n = threadid()

  @sync for i = 1:nao
     Threads.@spawn for j = 1:i
      half_loop1!(tmp, i, j, e_nao, p_nao, eri_ep_ao_ints, x1[ :, :, n], y1[ :, :,n], z1[ :,:, n], mo3_subset, mo4_subset, indx3, indx4) 
     end
  end

  @sync for k in 1:indx3
    Threads.@spawn for l in 1:indx4
      half_loop2!(tei_mo, k, l, e_nao, mo1_subset, mo2_subset, indx1, indx2, indx3, indx4, x2[:, :,n], y2[:,:,n], z2[:,:,n], tmp)
    end
  end

  return tei_mo

end

@btime mo_int4 = fast_conversion_general(nao, e_nao, p_nao, e_nocc, e_nvir, p_nocc, p_nvir, co_e, cv_e, co_n, cv_n, eri_ep_ao_ints)
mo_int4 = fast_conversion_general(nao, e_nao, p_nao, e_nocc, e_nvir, p_nocc, p_nvir, co_e, cv_e, co_n, cv_n, eri_ep_ao_ints)

mp2_ep = ep_energy(mo_int4, eia, ejb, p_nvir, p_nocc, e_nvir, e_nocc)
println(mp2_ep)
