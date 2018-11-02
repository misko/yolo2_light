#ifndef __DN_SERVER_FUNCTIONS__
#define __DN_SERVER_FUNCTIONS__
#include <pthread.h>
#include "additionally.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	TASK_NOT_READY,
	TASK_READY,
	TASK_DONE
} dn_task_status;

typedef struct {
	//input
	float * X;
	unsigned int number_of_images;

	//output	
	detection ** image_dets;
	int * nboxes;

	//synchronization
	pthread_mutex_t cv_mutex;
	pthread_cond_t cv;
	dn_task_status status;
} dn_gpu_task;

//get some tasks
int dn_enqueue(dn_gpu_task* t);
//dn_gpu_task * dn_dequeue(int * number_of_tasks); //private
dn_gpu_task * dn_create_task(int number_of_images);
void dn_destroy_task(dn_gpu_task*);

void dn_init_detector(int argc, char **argv);
void dn_close_detector();
detection ** dn_run_detector(float * data, unsigned int number_of_images, int * nboxes); 
float * dn_images_to_data(image * images, unsigned int number_of_images);
image * dn_resize_images(image * images, unsigned int number_of_images);
image * dn_load_images(const char ** filenames, unsigned int number_of_filenames);
void dn_free_images(image * images, unsigned int number_of_images);
char ** dn_get_names();
int dn_get_nboxes();
void dn_save_image(image im, char * filename);
void dn_draw_detections(image im,detection * dets, int nboxes);
void dn_free_detections(detection ** image_dets, int nboxes ,unsigned int number_of_images);

#ifdef __cplusplus
}
#endif
#endif
