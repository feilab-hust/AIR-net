import numpy as np
import tifffile
import torch

dtype = torch.cuda.FloatTensor
from core.utils.fft_conv import fft_conv,cfft_conv
from glbSettings import *

class PsfGenerator:

    def __init__(self, kbase, units, lam_detection, n, na_detection, n_threads=4):

        units = tuple(units)
        self.kbase= kbase.to(DEVICE)

        self.Nm,self.Nz, self.Ny, self.Nx = kbase.shape

        self.dy, self.dx = units

        self.na_detection = na_detection
        self.lam_detection = lam_detection
        self.n = n

        kx = torch.fft.fftfreq(self.Nx, self.dx).type(dtype).to(DEVICE)
        ky = torch.fft.fftfreq(self.Ny, self.dy).type(dtype).to(DEVICE)

        self.kcut = 1. * na_detection / self.lam_detection

        KY2, KX2 = torch.meshgrid(ky, kx, indexing="ij")
        KR2 = torch.hypot(KX2, KY2)

        self.krho = KR2 / self.kcut
        self.kphi = torch.arctan2(KY2, KX2)
        self.kmask2 = (KR2 <= self.kcut)

        self.myzifftn = lambda x: torch.fft.ifftn(x, dim=(1, 2))
        # self.myzfftn = lambda x: torch.fft.fftn(x, dim=(1, 2))
        self.zernike_cache = {i: self.zernike_polynomial(i) for i in range(55)}

    def zernike_polynomial(self, idx, normalized=False):

        R = self.krho
        THETA = self.kphi

        # see https://en.wikipedia.org/wiki/Zernike_polynomials
        # https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2016/03/ZP-Lecture-12.pdf

        # n = 0
        if idx == 0:
            F = torch.ones_like(R)

        # n = 1
        elif idx == 1:
            # Tip
            norm_factor = 2. if normalized else 1.
            F = norm_factor * torch.mul(R, torch.sin(THETA))
        elif idx == 2:
            # Tilt
            norm_factor = 2. if normalized else 1.
            F = norm_factor * torch.mul(R, torch.cos(THETA))

        # n = 2
        elif idx == 3:
            # Oblique astigmatism
            norm_factor = 6 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 2, torch.sin(2. * THETA))
        elif idx == 4:
            # Defocus
            norm_factor = 3 ** 0.5 if normalized else 1.
            F = norm_factor * (2. * R ** 2 - 1)
        elif idx == 5:
            # Vertical astigmatism
            norm_factor = 6 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 2, torch.cos(2. * THETA))

        # n = 3
        elif idx == 6:
            # Vertical trefoil
            norm_factor = 8 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 3, torch.sin(3. * THETA))
        elif idx == 7:
            # Vertical coma
            norm_factor = 8 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(3. * R ** 3 - 2. * R, torch.sin(THETA))
        elif idx == 8:
            # Horizontal coma
            norm_factor = 8 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(3. * R ** 3 - 2. * R, torch.cos(THETA))
        elif idx == 9:
            # Oblique trefoil
            norm_factor = 8 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 3, torch.cos(3. * THETA))

        # n = 4
        elif idx == 10:
            # Oblique quadrafoil
            norm_factor = 10 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 4, torch.sin(4. * THETA))
        elif idx == 11:
            # Oblique secondary astigmatism
            norm_factor = 10 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(4. * R ** 4 - 3. * R ** 2, torch.sin(2. * THETA))
        elif idx == 12:
            # Primary spherical
            norm_factor = 5 ** 0.5 if normalized else 1.
            F = norm_factor * (6. * R ** 4 - 6. * R ** 2 + torch.ones_like(R))
        elif idx == 13:
            # Vertical secondary astigmatism
            norm_factor = 10 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(4. * R ** 4 - 3. * R ** 2, torch.cos(2. * THETA))
        elif idx == 14:
            # Vertical quadrafoil
            norm_factor = 10 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 4, torch.cos(4. * THETA))

        # n = 5
        elif idx == 15:
            # Vertical pentafoil
            norm_factor = 12 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 5, torch.sin(5. * THETA))
        elif idx == 16:
            # Vertical secondary trefoil
            norm_factor = 12 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(5. * R ** 5 - 4. * R ** 3, torch.sin(3. * THETA))
        elif idx == 17:
            # Vertical secondary coma
            norm_factor = 12 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(10. * R ** 5 - 12. * R ** 3 + 3. * R, torch.sin(THETA))
        elif idx == 18:
            # Horizontal secondary coma
            norm_factor = 12 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(10. * R ** 5 - 12. * R ** 3 + 3. * R, torch.cos(THETA))
        elif idx == 19:
            # Oblique secondary trefoil
            norm_factor = 12 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(5. * R ** 5 - 4. * R ** 3, torch.cos(3. * THETA))
        elif idx == 20:
            # Oblique pentafoil
            norm_factor = 12 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 5, torch.cos(5. * THETA))

        # n = 6
        elif idx == 21:
            norm_factor = 14 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 6, torch.sin(6. * THETA))
        elif idx == 22:
            norm_factor = 14 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(6. * R ** 6 - 5. * R ** 4, torch.sin(4. * THETA))
        elif idx == 23:
            norm_factor = 14 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(15. * R ** 6 - 20. * R ** 4 + 6. * R ** 2, torch.sin(2. * THETA))
        elif idx == 24:
            norm_factor = 7 ** 0.5 if normalized else 1.
            F = norm_factor * (20. * R ** 6 - 30. * R ** 4 + 12. * R ** 2 - torch.ones_like(R))
        elif idx == 25:
            norm_factor = 14 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(15. * R ** 6 - 20. * R ** 4 + 6. * R ** 2, torch.cos(2. * THETA))
        elif idx == 26:
            norm_factor = 14 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(6. * R ** 6 - 5. * R ** 4, torch.cos(4. * THETA))
        elif idx == 27:
            norm_factor = 14 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 6, torch.cos(6. * THETA))

        # n = 7
        elif idx == 28:
            norm_factor = 16 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 7, torch.sin(7. * THETA))
        elif idx == 29:
            norm_factor = 16 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(7. * R ** 7 - 6. * R ** 5, torch.sin(5. * THETA))
        elif idx == 30:
            norm_factor = 16 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(21. * R ** 7 - 30. * R ** 5 + 10. * R ** 3, torch.sin(3. * THETA))
        elif idx == 31:
            norm_factor = 16 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(35. * R ** 7 - 60. * R ** 5 + 30. * R ** 3 - 4. * R, torch.sin(THETA))
        elif idx == 32:
            norm_factor = 16 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(35. * R ** 7 - 60. * R ** 5 + 30. * R ** 3 - 4. * R, torch.cos(THETA))
        elif idx == 33:
            norm_factor = 16 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(21. * R ** 7 - 30. * R ** 5 + 10. * R ** 3, torch.cos(3. * THETA))
        elif idx == 34:
            norm_factor = 16 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(7. * R ** 7 - 6. * R ** 5, torch.cos(5. * THETA))
        elif idx == 35:
            norm_factor = 16 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 7, torch.cos(7. * THETA))

        # n = 8
        elif idx == 36:
            norm_factor = 18 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 8, torch.sin(8. * THETA))
        elif idx == 37:
            norm_factor = 18 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(8. * R ** 8 - 7. * R ** 6, torch.sin(6. * THETA))
        elif idx == 38:
            norm_factor = 18 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(28. * R ** 8 - 42. * R ** 6 + 15. * R ** 4, torch.sin(4. * THETA))
        elif idx == 39:
            norm_factor = 18 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(56. * R ** 8 - 105. * R ** 6 + 60. * R ** 4 - 10. * R ** 2,
                                        torch.sin(2. * THETA))
        elif idx == 40:
            norm_factor = 9 ** 0.5 if normalized else 1.
            F = norm_factor * (70. * R ** 8 - 140. * R ** 6 + 90. * R ** 4 - 20. * R ** 2 + torch.ones_like(R))
        elif idx == 41:
            norm_factor = 18 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(56. * R ** 8 - 105. * R ** 6 + 60. * R ** 4 - 10. * R ** 2,
                                        torch.cos(2. * THETA))
        elif idx == 42:
            norm_factor = 18 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(28. * R ** 8 - 42. * R ** 6 + 15. * R ** 4, torch.cos(4. * THETA))
        elif idx == 43:
            norm_factor = 18 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(8. * R ** 8 - 7. * R ** 6, torch.cos(6. * THETA))
        elif idx == 44:
            norm_factor = 18 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 8, torch.cos(8. * THETA))

            # n = 9
        elif idx == 45:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 9, torch.sin(9. * THETA))
        elif idx == 46:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(9. * R ** 9 - 8. * R ** 7, torch.sin(7. * THETA))
        elif idx == 47:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(36. * R ** 9 - 56. * R ** 7 + 21. * R ** 5, torch.sin(5. * THETA))
        elif idx == 48:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(84. * R ** 9 - 168. * R ** 7 + 105. * R ** 5 - 20. * R ** 3,
                                        torch.sin(3. * THETA))
        elif idx == 49:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(126. * R ** 9 - 280. * R ** 7 + 210. * R ** 5 - 60. * R ** 3 + 5. * R,
                                        torch.sin(THETA))
        elif idx == 50:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(126. * R ** 9 - 280. * R ** 7 + 210. * R ** 5 - 60. * R ** 3 + 5. * R,
                                        torch.cos(THETA))
        elif idx == 51:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(84. * R ** 9 - 168. * R ** 7 + 105. * R ** 5 - 20. * R ** 3,
                                        torch.cos(3. * THETA))
        elif idx == 52:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(36. * R ** 9 - 56. * R ** 7 + 21. * R ** 5, torch.cos(5. * THETA))
        elif idx == 53:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(9. * R ** 9 - 8. * R ** 7, torch.cos(7. * THETA))
        elif idx == 54:
            norm_factor = 20 ** 0.5 if normalized else 1.
            F = norm_factor * torch.mul(R ** 9, torch.cos(9. * THETA))

        else:
            raise

        return F
    def masked_phase_array(self, phi):
        _phase = torch.zeros_like(self.krho)
        for j in range(len(phi)):
          _phase += phi[j] * self.zernike_cache[j + 3].type(dtype).to(DEVICE)  # 3 - 14
        return self.kmask2 * _phase


    def aberration_psf(self, phi):
        psf_calibration = torch.zeros(self.kbase.shape).to(DEVICE)
        for i in range(self.Nm):
            phi_temp = self.masked_phase_array(phi[i])
            ku = self.kbase[i] * torch.exp(2.j * torch.pi * phi_temp / self.lam_detection)
            res = self.myzifftn(ku)
            res=abs(torch.fft.ifftshift(res))**2
            res /= torch.sum(res)
            psf_calibration[i]=res

        return psf_calibration
