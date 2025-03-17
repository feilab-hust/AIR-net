
## FLFM_matlab
- The codes for FLFM PSF simulation and fourier light-field deconvolution with/without DAO, which are modified from olaf [1] and DAOSLIMIT[2].  <br>
Ref: \
[1] Stefanoiu, A., Page, J. & Lasser, T. "olaf: A flexible 3d reconstruction framework for light field microscopy". (2019). https://mediatum.ub.tum.de/1522002. \
[2].Wu, J., Lu, Z. Jiang, D. et al. "Iterative tomography with digital adaptive optics permits hour-long intravital observation of 3D subcellular dynamics at millisecond scale". Cell (2021).
 https://doi.org/10.1016/j.cell.2021.04.029

## Example usages

* FLFM PSF generation
    \
  *Note: Users can set the optical paramters in "PSF_simulation.m"*
  - Click "PSF_simulation.m" to generate PSF
  - Waiting for several minutes. PSF will be saved in forms of ".mat" in "./psf_matrix"
* Deconvolution for FLFM 3D reconstruction with/without DAO
  - Click "deconv_and_DAO.m",Setting the parameter DAO to 0 indicates the use of the traditional light field deconvolution algorithm, while setting it to 1 enables the DAO  algorithm.
  - Choose the  view stack of LF images 
  - Waiting for several minutes. Deconvolution results will be saved at the child folder "Deconv"
  \
  *Note: Users can set the iterative times in "deconv_and_DAO.m"*
