import numpy as np 


def mask_CS82(ra0,dec0):

	RA=(300, 60) ; DEC=(-2, 2)

	mask = ((ra0>RA[0])+(ra0<RA[1])) * ((dec0>DEC[0])*(dec0<DEC[1]))
	return mask

def mask_CFHT(ra0,dec0):

 	RA_W1=(28, 41) ; DEC_W1=(-13, -2)	# W1
 	RA_W2=(129, 139) ; DEC_W2=(-8, 1)	# W2
 	RA_W3=(207, 222) ; DEC_W3=(50, 59)	# W3
 	RA_W4=(329, 337) ; DEC_W4=(-2, 6) 	# W4

	mask_W1 = ((ra0>RA_W1[0])*(ra0<RA_W1[1])) * ((dec0>DEC_W1[0])*(dec0<DEC_W1[1]))
	mask_W2 = ((ra0>RA_W2[0])*(ra0<RA_W2[1])) * ((dec0>DEC_W2[0])*(dec0<DEC_W2[1]))
	mask_W3 = ((ra0>RA_W3[0])*(ra0<RA_W3[1])) * ((dec0>DEC_W3[0])*(dec0<DEC_W3[1]))
	mask_W4 = ((ra0>RA_W4[0])*(ra0<RA_W4[1])) * ((dec0>DEC_W4[0])*(dec0<DEC_W4[1]))
	return mask_W1, mask_W2, mask_W3, mask_W4

def mask_KiDS(ra0,dec0):

 	#RA_N1=(125, 240) ; DEC_N=(-5, 5)		# North
 	RA_N=(125, 240) ; DEC_N=(-5, 5)		# North
 	RA_S=(-30, 55) ; DEC_S=(-35, -25)	# South

	mask_N = ((ra0>RA_N[0])*(ra0<RA_N[1])) * ((dec0>DEC_N[0])*(dec0<DEC_N[1]))
	mask_S = ((ra0>RA_S[0])*(ra0<RA_S[1])) * ((dec0>DEC_S[0])*(dec0<DEC_S[1]))
	return mask_N, mask_S


mask_agn = (ra>125.)*(ra<240.)*(dec>-5.)*(dec<5.)*(zg>0.05)  # KiDS North
