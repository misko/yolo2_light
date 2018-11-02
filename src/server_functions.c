#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <time.h> 
#include <pthread.h> 
#include "box.h"
#include "server_functions.h"
//#include "pthread.h"
#include <errno.h>
#include "additionally.h"
#include <unistd.h>

static int batch_size;
static network net;
static char ** names;
static int obj_count;
static float thresh;
static int quantized;

pthread_mutex_t gpu_mutex; 

#define MAX_TASKS 16
#define QUEUE_SIZE 512
#define GPU_THREADS 4
#define GPU_WAIT_MS 100000 //0.1s
#define BATCH_SIZE 16

pthread_t gpu_threads[GPU_THREADS];
pthread_t gate_keeper_thread;;
dn_gpu_task ** work_queue[QUEUE_SIZE];
int work_queue_first_used=0;
int work_queue_used=0;
pthread_mutex_t work_queue_lock; 
pthread_cond_t cond_data_waiting;



int dn_enqueue(dn_gpu_task* t) {
	if (t->number_of_images<=0) {
		return -1;
	}
	int ret = pthread_mutex_lock(&work_queue_lock);
	if (ret!=0) {
		fprintf(stderr,"CRITICAL MUTEX ERROR\n");
		exit(1);
	}

	//CRITICAL REGION
	
	if (work_queue_used==QUEUE_SIZE) {
		fprintf(stdout,"WORK QUEUE IS FULL!\n");
		pthread_mutex_unlock(&work_queue_lock);
		return -1;
	}

	work_queue[(work_queue_first_used+work_queue_used)%QUEUE_SIZE]=t;

	work_queue_used=work_queue_used+1;
	//fprintf(stderr,"QUEUE HAS %d\n",work_queue_used);	
	//END CRITICAL REGION
	//pthread_cond_broadcast(&cond_data_waiting);
	
	pthread_mutex_unlock(&work_queue_lock);
	return 0;
}

void * dn_gate_keeper(void * x) {
	int waits=0;
	while (1) {
		usleep(GPU_WAIT_MS/2);
		if (work_queue_used>0) {
			waits++;
		}
		if (waits>=4) {
			pthread_cond_signal(&cond_data_waiting);
			waits=-1;
		}
	}
}

dn_gpu_task * dn_dequeue(int * number_of_tasks) {
	int ret = pthread_mutex_lock(&work_queue_lock);
	if (ret!=0) {
		fprintf(stderr,"CRITICAL MUTEX ERROR\n");
		exit(1);
	}


	//wait from the gate keeper
	pthread_cond_wait(&cond_data_waiting, &work_queue_lock);
	while (work_queue_used<=0) {
		pthread_cond_wait(&cond_data_waiting, &work_queue_lock);
	}

	//CRITICAL REGION
	int tasks_to_take=0;
	int used_in_this_batch=0;
	while (tasks_to_take<MAX_TASKS && (work_queue_used-tasks_to_take)>0 && used_in_this_batch<2*batch_size) {
		dn_gpu_task * next_task = work_queue[(work_queue_first_used+tasks_to_take)%QUEUE_SIZE];
		//add this one to the batch
		used_in_this_batch+=next_task->number_of_images;
		tasks_to_take++;
	}

	dn_gpu_task ** my_tasks = (dn_gpu_task**)malloc(sizeof(dn_gpu_task)*tasks_to_take);
	for (int i=0; i<tasks_to_take; i++) {
		//fprintf(stdout,"TAKING TASKS %d\n",(work_queue_first_used+i)%QUEUE_SIZE);
		my_tasks[i]=work_queue[(work_queue_first_used+i)%QUEUE_SIZE];
	}
	*number_of_tasks=tasks_to_take;
	work_queue_first_used=(work_queue_first_used+tasks_to_take)%QUEUE_SIZE;
	work_queue_used-=tasks_to_take;
	
	//END CRITICAL REGION
	
	pthread_mutex_unlock(&work_queue_lock);

	return my_tasks;
}


dn_gpu_task * dn_create_task(int number_of_images) {
	dn_gpu_task * t = (dn_gpu_task*)calloc(1,sizeof(dn_gpu_task));
	if (t==NULL) {
		fprintf(stderr,"out of memory\n");
		exit(1);
	}
	t->image_dets=(detection**)calloc(1,sizeof(detection*)*number_of_images);
	if (t->image_dets==NULL) {
		fprintf(stderr,"out of memory\n");
		exit(1);
	}
	t->number_of_images=number_of_images;
	t->nboxes=(int*)calloc(1,sizeof(int)*number_of_images);
	if (t->nboxes==NULL) {
		fprintf(stderr,"out of memory\n");
	}
	//t->X=NULL; //done by calloc
	if (pthread_mutex_init(&(t->cv_mutex), NULL) != 0)  { 
		printf("\n mutex init has failed\n"); 
		exit(1); 
	} 
	if ( pthread_cond_init(&(t->cv), NULL)!=0) {
		printf("\n cond init has failed\n"); 
		exit(1); 
	}
	t->status=TASK_NOT_READY;
	return t;
}

void dn_destroy_task(dn_gpu_task*t) {
	pthread_mutex_destroy(&(t->cv_mutex));
	pthread_cond_destroy(&(t->cv));
	free(t->nboxes);
	free(t->image_dets);
	free(t);
}



// get prediction boxes: yolov2_forward_network.c
void get_region_boxes_cpu(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);

typedef struct detection_with_class {
    detection det;
    // The most probable class id: the best class index in this->prob.
    // Is filled temporary when processing results, otherwise not initialized
    int best_class;
} detection_with_class;

// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num)
{
    int selected_num = 0;
    detection_with_class* result_arr = calloc(dets_num, sizeof(detection_with_class));
    int i;
    for (i = 0; i < dets_num; ++i) {
        int best_class = -1;
        float best_class_prob = thresh;
        int j;
        for (j = 0; j < dets[i].classes; ++j) {
            if (dets[i].prob[j] > best_class_prob) {
                best_class = j;
                best_class_prob = dets[i].prob[j];
            }
        }
        if (best_class >= 0) {
            result_arr[selected_num].det = dets[i];
            result_arr[selected_num].best_class = best_class;
            ++selected_num;
        }
    }
    if (selected_detections_num)
        *selected_detections_num = selected_num;
    return result_arr;
}

// compare to sort detection** by bbox.x
int compare_by_lefts(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    const float delta = (a->det.bbox.x - a->det.bbox.w / 2) - (b->det.bbox.x - b->det.bbox.w / 2);
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

// compare to sort detection** by best_class probability
int compare_by_probs(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    float delta = a->det.prob[a->best_class] - b->det.prob[b->best_class];
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

char ** dn_get_names() {
	return names;
}

void draw_detections_v3(image im, detection *dets, int num, float thresh, char **names, int classes, int ext_output)
{
    int selected_detections_num;
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num);

    // text output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
    int i;
    /*
    for (i = 0; i < selected_detections_num; ++i) {
        const int best_class = selected_detections[i].best_class;
        printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
        if (ext_output)
            printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
            (selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w,
                (selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h,
                selected_detections[i].det.bbox.w*im.w, selected_detections[i].det.bbox.h*im.h);
        else
            printf("\n");
        int j;
        for (j = 0; j < classes; ++j) {
            if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
                printf("%s: %.0f%%\n", names[j], selected_detections[i].det.prob[j] * 100);
            }
        }
    }*/

    // image output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
    for (i = 0; i < selected_detections_num; ++i) {
        int width = im.h * .006;
        if (width < 1)
            width = 1;

        //printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
        int offset = selected_detections[i].best_class * 123457 % classes;
        float red = get_color(2, offset, classes);
        float green = get_color(1, offset, classes);
        float blue = get_color(0, offset, classes);
        float rgb[3];

        //width = prob*20+2;

        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
        box b = selected_detections[i].det.bbox;
        //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

        int left = (b.x - b.w / 2.)*im.w;
        int right = (b.x + b.w / 2.)*im.w;
        int top = (b.y - b.h / 2.)*im.h;
        int bot = (b.y + b.h / 2.)*im.h;

        if (left < 0) left = 0;
        if (right > im.w - 1) right = im.w - 1;
        if (top < 0) top = 0;
        if (bot > im.h - 1) bot = im.h - 1;

        draw_box_width(im, left, top, right, bot, width, red, green, blue);
    }
    free(selected_detections);
}


image * dn_load_images(const char ** filenames, unsigned int number_of_filenames) {
	if (number_of_filenames<=0) {
		return NULL;
	}
	image * images = (image*)malloc(sizeof(image)*number_of_filenames);
	if (images==NULL) {
		fprintf(stderr,"MALLOC FIALED\n");
		return NULL;
	}
	for (int n=0; n<number_of_filenames; n++) {
        	images[n] = load_image(filenames[n], 0, 0, 3);            // image.c
	}
	return images;
}

image * dn_resize_images(image * images, unsigned int number_of_images) {
	if (number_of_images<=0) {
		return NULL;
	}
	image * resized_images = (image*)malloc(sizeof(image)*number_of_images);
	if (resized_images==NULL) {
		fprintf(stderr,"MALLOC FIALED RESIZED\n");
		return NULL;
	}
	for (int n=0; n<number_of_images; n++) {
        	resized_images[n] = resize_image(images[n], net.w, net.h);    // image.c
	}
	return resized_images;
}

float * dn_images_to_data(image * images, unsigned int number_of_images) {
	if (number_of_images<=0) {
		return NULL;
	}
	size_t image_size = images[0].w*images[0].h*images[0].c;
	float * data = (float*)malloc(sizeof(float)*image_size*number_of_images);
	for (int n=0; n<number_of_images; n++) {
		memcpy(data+n*image_size,images[n].data,image_size*sizeof(float));
	}
	return data;
}

void dn_free_images(image * images, unsigned int number_of_images) {
	for (int n=0; n<number_of_images; n++) {
		free_image(images[n]);                    // image.c
	}
}

// --------------- Detect on the Image ---------------

void dn_draw_detections(image im,detection * dets, int nboxes) {
	draw_detections_v3(im, dets, nboxes, thresh, names, 6, 1);
}

void dn_save_image(image im, char * filename) {
	save_image_png(im, filename);    // image.c
}

void dn_free_detections(detection ** image_dets, int nboxes, unsigned int number_of_images) {
	for (int n=0; n<number_of_images; n++) {
		free_detections(image_dets[n],nboxes);
	}
}
// Detect on Image: this function uses other functions not from this file
void * dn_detector_worker(void * x)  {

    clock_t time;
    float nms = .4;

    //set_batch_network(&net, batch_size);                    // network.c
    size_t image_size = net.w*net.h*3;

    //get input buffer ready
    float *X = (float *)malloc(sizeof(float)*image_size*batch_size);
    if (X==NULL) {
	fprintf(stderr,"Failed to malloc memory for image batch input buffer\n");
	exit(1);
    }

    detection *** image_dets = (detection***)malloc(sizeof(detection**)*batch_size);
    if (image_dets==NULL) {
	fprintf(stderr,"Failed to malloc memory for image batch input buffer\n");
	exit(1);
    }
    int ** batch_nboxes = (int**)malloc(sizeof(int*)*batch_size);
    if (batch_nboxes==NULL) {
	fprintf(stderr,"FAILED MALLOC\n");
	exit(1);
    }	

    fprintf(stderr,"SERVER THREAD READY image_dets %p\n",image_dets);
    while (1) {
	//lets get a batch
	int number_of_tasks=0;
    	//fprintf(stderr,"SERVER THREAD DEQUEUE WAIT\n");
	dn_gpu_task ** my_tasks = dn_dequeue(&number_of_tasks);	
	if (number_of_tasks==0) {
		fprintf(stderr,"that was weird\n");
		continue;
	}
    	//fprintf(stderr,"SERVER THREAD DEQUEUED\n");

	//find out how many images we have
	int images_to_process=0;
	for (int task_idx=0; task_idx<number_of_tasks; task_idx++) {
		const dn_gpu_task * t = my_tasks[task_idx];
		images_to_process+=t->number_of_images;
	}

	int last_task=0;
	int last_image=0;
	int images_processed=0;

    	//fprintf(stderr,"SERVER THREAD PROCESSING\n");
	while (images_processed<images_to_process) {
		//zero the input buffer
		memset(X,0,sizeof(float)*image_size*batch_size);
		memset(image_dets,0,sizeof(detection**)*batch_size);
		memset(batch_nboxes,0,sizeof(int*)*batch_size);
		
		//find out how many images in this specific batch
		int images_in_this_batch=(images_to_process-images_processed);
		if (images_in_this_batch>batch_size) {
			images_in_this_batch=batch_size;
		}
    		fprintf(stderr,"SERVER THREAD PROCESSING - images_in_this_batch %d\n",images_in_this_batch);

		//now copy the input data to our buffer
		for (int i=0; i<images_in_this_batch;) {
			dn_gpu_task * t = my_tasks[last_task]; //get the task at hand
			//fprintf(stderr,"CHECKING TASK %d %p , image_dets %p\n",i,t,t->image_dets);
			int images_to_load=0;
			if ((t->number_of_images-last_image)<=(images_in_this_batch-i)) {
				//load the rest of the task
				images_to_load = (t->number_of_images-last_image);
			} else {
				//load only part of it
				images_to_load = images_in_this_batch - i;
			}	
			//fprintf(stderr,"IMAGES TO LOAD %d\n",images_to_load);
			//copy the inputs and the output pointers
			memcpy(X+i*image_size,t->X+last_image*image_size,images_to_load*image_size*sizeof(float));
			for (int j=0; j<images_to_load; j++) {
				//fprintf(stderr,"image_dets[%d]=%p\n",i+j,t->image_dets+last_image+i+j);
				image_dets[i+j]=t->image_dets+last_image+j;
				batch_nboxes[i+j]=t->nboxes+last_image+j;
			}

			last_image+=images_to_load;
			if (last_image==t->number_of_images) {
				last_task++;
				last_image=0;
			}
			i+=images_to_load;
		}

		//lets try to grab that network gpu
		pthread_mutex_lock(&gpu_mutex);
		//fprintf(stderr,"GOT TO THE GPU!\n");
		//run the network
#ifdef GPU
		if (quantized) {
			network_predict_gpu_cudnn_quantized(net, X);    // quantized works only with Yolo v2
			//nms = 0.2;
		}
		else {
			network_predict_gpu_cudnn(net, X);
		}
#else
#ifdef OPENCL
		network_predict_opencl(net, X);
#else
		if (quantized) {
			network_predict_quantized(net, X);    // quantized works only with Yolo v2
			nms = 0.2;
		}
		else {
			network_predict_cpu(net, X);
		}
#endif
#endif

        	layer l = net.layers[net.n - 1];
		//fprintf(stderr,"GOT TO THE GPU! - DONE\n");
		const float hier_thresh = 0.5;
		const int ext_output = 1, letterbox = 0;
		for (int b=0; b<images_in_this_batch; b++) {
			*image_dets[b] = get_network_boxes(&net, net.w, net.h, thresh, hier_thresh, 0, 1, batch_nboxes[b], letterbox, b );
		}
		pthread_mutex_unlock(&gpu_mutex);

		for (int b=0; b<images_in_this_batch; b++) {
			if (nms) do_nms_sort(*image_dets[b], *batch_nboxes[b], l.classes, nms);
		}
		
		images_processed+=images_in_this_batch;
	}
	for (int task_idx=0; task_idx<number_of_tasks; task_idx++) {
		dn_gpu_task * t = my_tasks[task_idx];
		t->status=TASK_DONE;

		int ret = pthread_mutex_lock(&t->cv_mutex);
		if (ret!=0) {
			fprintf(stderr,"CRITICAL MUTEX ERROR\n");
			exit(1);
		}
		pthread_cond_broadcast(&(t->cv));
		pthread_mutex_unlock(&t->cv_mutex);

	}

    }

/*
    detection ** image_dets = (detection**)malloc(sizeof(detection*)*number_of_images);
    if (image_dets==NULL) {
	fprintf(stderr,"FAILED TO MALLOC IMAGE DETS\n");
	return NULL;
    }
    fprintf(stderr,"RUNNING IN BATCH MODE\n");
    int processing_index=0;
    while (processing_index<number_of_images) {
        layer l = net.layers[net.n - 1];

	//float *X = sized.data;
	float *X = (float *)calloc(1,sizeof(float)*image_size*batch_size);
	int images_in_this_batch=batch_size;
	if (number_of_images-processing_index<batch_size) {
		images_in_this_batch=number_of_images-processing_index;
	}
	memcpy(X,data+image_size*processing_index,images_in_this_batch*image_size*sizeof(float));
        time = clock();
	//network_predict(net, X);
	//
	//
	//lets try to grab that network gpu
	pthread_mutex_lock(&network_mutex);

	struct timespec max_wait = {0, 0};
	clock_gettime(CLOCK_REALTIME, &max_wait);
        max_wait.tv_sec +=5;;
	while (gpu_busy==1) {
		int err = pthread_cond_timedwait(&cond_gpu_busy, &network_mutex, &max_wait);
		if (err == ETIMEDOUT) {
			fprintf(stderr,"TIMEOUT!!\n");
			free(image_dets);
			int nboxes=dn_get_nboxes();
			for (int i=0; i<processing_index; i++) {
				free_detections(image_dets[i], nboxes);
			}
			return NULL;
		}
	}
	gpu_busy=1;
	pthread_mutex_unlock(&network_mutex);

	//run the network
#ifdef GPU
        if (quantized) {
            network_predict_gpu_cudnn_quantized(net, X);    // quantized works only with Yolo v2
                                                            //nms = 0.2;
        }
        else {
            network_predict_gpu_cudnn(net, X);
        }
#else
#ifdef OPENCL
        network_predict_opencl(net, X);
#else
        if (quantized) {
            network_predict_quantized(net, X);    // quantized works only with Yolo v2
            nms = 0.2;
        }
        else {
            network_predict_cpu(net, X);
        }
#endif
#endif
	gpu_busy=0;
	pthread_cond_broadcast(&cond_gpu_busy); 
        printf("%s: Predicted in %f seconds.\n", "X", (float)(clock() - time) / CLOCKS_PER_SEC); //sec(clock() - time));

	for (int b=0; b<images_in_this_batch; b++) {
		float hier_thresh = 0.5;
		int ext_output = 1, letterbox = 0, nboxes = 0;
		detection *dets = get_network_boxes(&net, net.w, net.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox, b );
		if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
		//draw_detections_v3(resized_images[i], dets, nboxes, thresh, names, l.classes, ext_output);


		image_dets[processing_index+b]=dets;

		//free_image(resized_images[i]);                    // image.c
		//free_detections(dets, nboxes);
	}

	processing_index+=images_in_this_batch;
	free(X);
    }
    return image_dets;*/
}
// Detect on Image: this function uses other functions not from this file
detection ** dn_run_detector(float * data, unsigned int number_of_images, int * result_nboxes) 
{

    clock_t time;
    float nms = .4;

    //set_batch_network(&net, batch_size);                    // network.c
    size_t image_size = net.w*net.h*3;


    detection ** image_dets = (detection**)malloc(sizeof(detection*)*number_of_images);
    if (image_dets==NULL) {
	fprintf(stderr,"FAILED TO MALLOC IMAGE DETS\n");
	return NULL;
    }
    fprintf(stderr,"RUNNING IN BATCH MODE\n");
    int processing_index=0;
    while (processing_index<number_of_images) {
        layer l = net.layers[net.n - 1];

	//float *X = sized.data;
	float *X = (float *)calloc(1,sizeof(float)*image_size*batch_size);
	int images_in_this_batch=batch_size;
	if (number_of_images-processing_index<batch_size) {
		images_in_this_batch=number_of_images-processing_index;
	}
	memcpy(X,data+image_size*processing_index,images_in_this_batch*image_size*sizeof(float));

	//lets try to grab that network gpu
	pthread_mutex_lock(&gpu_mutex);

	//run the network
#ifdef GPU
        if (quantized) {
            network_predict_gpu_cudnn_quantized(net, X);    // quantized works only with Yolo v2
                                                            //nms = 0.2;
        }
        else {
            network_predict_gpu_cudnn(net, X);
        }
#else
#ifdef OPENCL
        network_predict_opencl(net, X);
#else
        if (quantized) {
            network_predict_quantized(net, X);    // quantized works only with Yolo v2
            nms = 0.2;
        }
        else {
            network_predict_cpu(net, X);
        }
#endif
#endif
        printf("%s: Predicted in %f seconds.\n", "X", (float)(clock() - time) / CLOCKS_PER_SEC); //sec(clock() - time));

	for (int b=0; b<images_in_this_batch; b++) {
		float hier_thresh = 0.5;
		int ext_output = 1, letterbox = 0, nboxes = 0;
		detection *dets = get_network_boxes(&net, net.w, net.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox, b );
		if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
		//draw_detections_v3(resized_images[i], dets, nboxes, thresh, names, l.classes, ext_output);

		/*char buffer[256];
		sprintf(buffer,"predictions_%d",processing_index+b);
		image resized_image ; 
		resized_image.w=net.w;
		resized_image.h=net.h;
		resized_image.c=3;
		resized_image.data=data+image_size*(processing_index+b);
		save_image_png(resized_image, buffer);    // image.c*/

		image_dets[processing_index+b]=dets;
		result_nboxes[processing_index+b]=nboxes;

		//free_image(resized_images[i]);                    // image.c
		//free_detections(dets, nboxes);
	}
	pthread_mutex_unlock(&gpu_mutex);

	processing_index+=images_in_this_batch;
	free(X);
    }
    return image_dets;
}



// --------------- Detect on the Video ---------------

// get command line parameters and load objects names
void dn_init_detector(int argc, char **argv)
{
	if (pthread_mutex_init(&work_queue_lock, NULL) != 0) 
	{ 
		printf("\n mutex init has failed\n"); 
		exit(1); 
	} 
	if (pthread_mutex_init(&gpu_mutex, NULL) != 0) 
	{ 
		printf("\n mutex init has failed\n"); 
		exit(1); 
	} 
	if ( pthread_cond_init(&cond_data_waiting, NULL)!=0)
	{
		printf("\n cond init has failed\n"); 
		exit(1); 

	}


    int gpu_index = find_int_arg(argc, argv, "-i", 0);  //  gpu_index = 0;
#ifndef GPU
    gpu_index = -1;
#else
    if (gpu_index >= 0) {
        cuda_set_device(gpu_index);
    }
#endif
#ifdef OPENCL
    ocl_initialize();
#endif
    

    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    thresh = find_float_arg(argc, argv, "-thresh", .24);
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int quantized = find_arg(argc, argv, "-quantized");
    int input_calibration = find_int_arg(argc, argv, "-input_calibration", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [demo/test/] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int clear = 0;                // find_arg(argc, argv, "-clear");

    char *obj_names = argv[3];    // char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6] : 0;

    // load object names
    names = calloc(10000, sizeof(char *));
    obj_count = 0;
    FILE* fp;
    char buffer[255];
    fp = fopen(obj_names, "r");
    if (fp==NULL) {
	fprintf(stderr,"Filaed to open file %s\n",obj_names);
exit(1);
	}
    while (fgets(buffer, 255, (FILE*)fp)) {
        names[obj_count] = calloc(strlen(buffer)+1, sizeof(char));
        strcpy(names[obj_count], buffer);
        names[obj_count][strlen(buffer) - 1] = '\0'; //remove newline
        ++obj_count;
    }
    fclose(fp);
    int classes = obj_count;

    batch_size=BATCH_SIZE;

    net = parse_network_cfg(cfg, batch_size, quantized);    // parser.c
    if (weights) {
        load_weights_upto_cpu(&net, weights, net.n);    // parser.c
    }
    srand(2222222);
    yolov2_fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (quantized) {
	    printf("\n\n Quantinization! \n\n");
	    quantinization_and_get_multipliers(net);
    }

    //test_detector_cpu_batch(names, cfg, weights, filename, thresh, quantized, dont_show);
    //
    //launch GPU workers
    for (int i=0; i<GPU_THREADS; i++) {
	    if(pthread_create(gpu_threads+i, NULL, dn_detector_worker, NULL)) {

		    fprintf(stderr, "Error creating thread\n");
		    exit(1);

	    }
    }
    if(pthread_create(&gate_keeper_thread, NULL, dn_gate_keeper, NULL)){
		    fprintf(stderr, "Error creating thread\n");
		    exit(1);

    }
}


void dn_close_detector() {
   //TODO 

    int i;
    for (i = 0; i < obj_count; ++i) free(names[i]);
    free(names);
}


