#ifndef DETECT_H
#define DETECT_H
#include "embARC.h"
#include "embARC_debug.h"
#include "embARC_error.h"

#define frame_size	(64*64*3)
#define input_size	(36*36*3)
#define frame_w		(64)
#define frame_h		(64)
#define window_w	(36)
#define window_h	(36)
#define window_num	(window_w * window_h)	
#define stride_w	(4)
#define stride_h	(4)


void face_detect(uint8_t frame_buffer[frame_size]);


#endif