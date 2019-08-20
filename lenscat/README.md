# LensCat module description

## Catalogue Description

Some details and reference links for each catalogue...


## How to use it




## Catalogue Format

### KiDS

Files are in HDF5 format. Tables are grouped as follows:

 - coordinates
    - ID
    - RAJ2000
    - DECJ2000
    - Z_B
    - Z_B_MIN
    - Z_B_MAX
    - T_B
    - ODDS
    - SG_FLAG

 - lensing
    - ID
    - PSF_Q11
    - PSF_Q12
    - PSF_Q22
    - PSF_Strehl_ratio
    - PSF_e1
    - PSF_e1_exp1
    - PSF_e1_exp2
    - PSF_e1_exp3
    - PSF_e1_exp4
    - PSF_e1_exp5
    - PSF_e2
    - PSF_e2_exp1
    - PSF_e2_exp2
    - PSF_e2_exp3
    - PSF_e2_exp4
    - PSF_e2_exp5
    - bias_corrected_scalelength
    - bulge_fraction
    - contamination_radius
    - e1
    - e2
    - m
    - weight
    - fitclass
    - model_SNratio
    - model_flux
    - n_exposures_used
    - pixel_SNratio

 - magnitudes
    - ID
    - CLASS_STAR
    - FLUX_RADIUS
    - FWHM_IMAGE
    - FWHM_WORLD
    - Flag
    - KRON_RADIUS
    - MAGERR_g
    - MAGERR_i
    - MAGERR_r
    - MAGERR_u
    - MAG_LIM_g
    - MAG_LIM_i
    - MAG_LIM_r
    - MAG_LIM_u
    - MAG_g
    - MAG_i
    - MAG_r
    - MAG_u
    - Xpos
    - Ypos
    - ZPT_offset

 - others
    - ID
    - KIDS_TILE
    - MASK
    - Patch
    - SeqNr
    - THELI_NAME
