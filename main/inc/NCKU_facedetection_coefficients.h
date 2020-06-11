#ifndef	_NCKU_FD_COEFFICIENTS_H_
#define _NCKU_FD_COEFFICIENTS_H_

#include "mli_config.h"

#include "NCKU_facedetection.h"

//Define Fix-Point Calculate Function
#define QMN(type, fraq, val)   (type)(val * (1u << (fraq)) + ((val >= 0)? 0.5f: -0.5f))
#define FRQ_BITS(int_part, el_type) ((sizeof(el_type)*8)-int_part-1)

typedef int16_t w_type;
typedef int16_t d_type;

#define _Wdata_attr __attribute__((section(".mli_model")))
#define _W __xy _Wdata_attr

// Bank X (XCCM) attribute
#define __Xdata_attr __attribute__((section(".Xdata")))
#define _X __xy __Xdata_attr

// Bank Y (YCCM) attribute
#define __Ydata_attr __attribute__((section(".Ydata")))
#define _Y __xy __Ydata_attr

// Bank Z (DCCM) attribute
#define __Zdata_attr __attribute__((section(".Zdata")))
#define _Z __xy __Zdata_attr

//-----------------------------------------------
//Define Integer Bits per Layer
//-----------------------------------------------
#define IMAGE_INT		(8)

#define RESIZE_W_INT	(-3)
#define RESIZE_B_INT	(0)
#define RESIZE_OUT_INT	(8)

#define CONV0_W_INT		(-7)
#define CONV0_B_INT		(0)
#define CONV0_OUT_INT	(1)

#define CONV1_W_INT   	(2)
#define CONV1_B_INT   	(1)
#define CONV1_P_INT	  	(1)
#define CONV1_OUT_INT 	(3)
#define CONV1_POUT_INT 	(3)
					   
#define CONV2_W_INT   	(1)
#define CONV2_B_INT   	(2)
#define CONV2_P_INT	  	(0)
#define CONV2_OUT_INT 	(3)
#define CONV2_POUT_INT 	(2)
						
#define CONV3_W_INT   	(0)
#define CONV3_B_INT   	(1)
#define CONV3_P_INT	  	(0)
#define CONV3_OUT_INT 	(2)
#define CONV3_POUT_INT 	(2)
						
#define CONV4_W_INT   	(0)
#define CONV4_B_INT   	(-10)
#define CONV4_OUT_INT 	(3)
					  
//(Co, Ci, h, w)
//-----------------------------------------------
//Shape and Fractional bits per layer definitions
//-----------------------------------------------
//Input_image
//-----------------------------------------------
#define IMAGE_SHAPE {3,36,36}
#define IMAGE_ELEMENTS (3*36*36)
#define IMAGE_RANK (3)

#define IMAGE_FRAQ		(FRQ_BITS(IMAGE_INT, w_type))
#define IQ(val)		QMN(w_type, IMAGE_FRAQ, val)
//Resize
//-----------------------------------------------
#define RESIZE_W_SHAPE {3,3,3,3}
#define RESIZE_W_ELEMENTS (3*3*3*3)
#define RESIZE_W_RANK (4)

#define RESIZE_W_FRAQ   (FRQ_BITS(RESIZE_W_INT, w_type))
#define LR_WQ(val)   QMN(w_type, RESIZE_W_FRAQ, val)

#define RESIZE_B_ELEMENTS (3)
#define RESIZE_B_SHAPE {3}
#define RESIZE_B_RANK (3)

#define RESIZE_B_FRAQ   (FRQ_BITS(RESIZE_B_INT, w_type))
#define LR_BQ(val)   QMN(w_type, RESIZE_B_FRAQ, val)

#define RESIZE_OUT_FRAQ (FRQ_BITS(RESIZE_OUT_INT, d_type))
//CONV0
//-----------------------------------------------
#define CONV0_W_SHAPE {3,3,1,1}
#define CONV0_W_ELEMENTS (3*3*1*1)
#define CONV0_W_RANK (4)

#define CONV0_W_FRAQ   (FRQ_BITS(CONV0_W_INT, w_type))
#define L0_WQ(val)   QMN(w_type, CONV0_W_FRAQ, val)

#define CONV0_B_ELEMENTS (3)
#define CONV0_B_SHAPE {3}
#define CONV0_B_RANK (3)

#define CONV0_B_FRAQ   (FRQ_BITS(CONV0_B_INT, w_type))
#define L0_BQ(val)   QMN(w_type, CONV0_B_FRAQ, val)

#define CONV0_OUT_FRAQ (FRQ_BITS(CONV0_OUT_INT, d_type))
//CONV1
//-----------------------------------------------
#define CONV1_W_SHAPE {10,3,3,3}
#define CONV1_W_ELEMENTS (10*3*3*3)
#define CONV1_W_RANK (4)

#define CONV1_W_FRAQ   (FRQ_BITS(CONV1_W_INT, w_type))
#define L1_WQ(val)   QMN(w_type, CONV1_W_FRAQ, val)

#define CONV1_B_ELEMENTS (10)
#define CONV1_B_SHAPE {10}
#define CONV1_B_RANK (1)

#define CONV1_B_FRAQ   (FRQ_BITS(CONV1_B_INT, w_type))
#define L1_BQ(val)   QMN(w_type, CONV1_B_FRAQ, val)

#define CONV1_OUT_FRAQ (FRQ_BITS(CONV1_OUT_INT, d_type))

#define CONV1_P_ELEMENTS (10)
#define CONV1_P_SHAPE {10}
#define CONV1_P_RANK (1)

#define CONV1_P_FRAQ   (FRQ_BITS(CONV1_P_INT, w_type))
#define L1_PQ(val)   QMN(w_type, CONV1_P_FRAQ, val)

#define CONV1_POUT_FRAQ (FRQ_BITS(CONV1_POUT_INT, d_type)) 
//CONV2
//-----------------------------------------------
#define CONV2_W_SHAPE {16,10,3,3}
#define CONV2_W_ELEMENTS (16*10*3*3)
#define CONV2_W_RANK (4)

#define CONV2_W_FRAQ   (FRQ_BITS(CONV2_W_INT, w_type))
#define L2_WQ(val)   QMN(w_type, CONV2_W_FRAQ, val)

#define CONV2_B_ELEMENTS (16)
#define CONV2_B_SHAPE {16}
#define CONV2_B_RANK (1)

#define CONV2_B_FRAQ   (FRQ_BITS(CONV2_B_INT, w_type))
#define L2_BQ(val)   QMN(w_type, CONV2_B_FRAQ, val)

#define CONV2_OUT_FRAQ (FRQ_BITS(CONV2_OUT_INT, d_type))

#define CONV2_P_ELEMENTS (16)
#define CONV2_P_SHAPE {16}
#define CONV2_P_RANK (1)

#define CONV2_P_FRAQ   (FRQ_BITS(CONV2_P_INT, w_type))
#define L2_PQ(val)   QMN(w_type, CONV2_P_FRAQ, val)

#define CONV2_POUT_FRAQ (FRQ_BITS(CONV2_POUT_INT, d_type)) 

//CONV3
//-----------------------------------------------
#define CONV3_W_SHAPE {32,16,3,3}
#define CONV3_W_ELEMENTS (32*16*3*3)
#define CONV3_W_RANK (4)

#define CONV3_W_FRAQ   (FRQ_BITS(CONV3_W_INT, w_type))
#define L3_WQ(val)   QMN(w_type, CONV3_W_FRAQ, val)

#define CONV3_B_ELEMENTS (32)
#define CONV3_B_SHAPE {32}
#define CONV3_B_RANK (1)

#define CONV3_B_FRAQ   (FRQ_BITS(CONV3_B_INT, w_type))
#define L3_BQ(val)   QMN(w_type, CONV3_B_FRAQ, val)

#define CONV3_OUT_FRAQ (FRQ_BITS(CONV3_OUT_INT, d_type))

#define CONV3_P_ELEMENTS (32)
#define CONV3_P_SHAPE {32}
#define CONV3_P_RANK (1)

#define CONV3_P_FRAQ   (FRQ_BITS(CONV3_P_INT, w_type))
#define L3_PQ(val)   QMN(w_type, CONV3_P_FRAQ, val)

#define CONV3_POUT_FRAQ (FRQ_BITS(CONV3_POUT_INT, d_type)) 
//CONV4
//-----------------------------------------------
#define CONV4_W_SHAPE {2,32,1,1}
#define CONV4_W_ELEMENTS (2*32*1*1)
#define CONV4_W_RANK (4)

#define CONV4_W_FRAQ   (FRQ_BITS(CONV4_W_INT, w_type))
#define L4_WQ(val)   QMN(w_type, CONV4_W_FRAQ, val)

#define CONV4_B_ELEMENTS (2)
#define CONV4_B_SHAPE {2}
#define CONV4_B_RANK (1)

#define CONV4_B_FRAQ   (FRQ_BITS(CONV4_B_INT, w_type))
#define L4_BQ(val)   QMN(w_type, CONV4_B_FRAQ, val)

#define CONV4_OUT_FRAQ (FRQ_BITS(CONV4_OUT_INT, d_type))


#endif