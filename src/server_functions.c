#include <stdio.h>
#include <stdlib.h>

#include "box.h"
#include "pthread.h"

#include "additionally.h"

static int batch_size;
static network net;
static char ** names;
static int obj_count;
static float thresh;
static int quantized;

int get_nboxes() {
	return num_detections(&net,thresh);
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

char ** get_names() {
	return names;
}

void draw_detections_v3(image im, detection *dets, int num, float thresh, char **names, int classes, int ext_output)
{
    int selected_detections_num;
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num);

    // text output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
    int i;
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
    }

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


image * load_images(const char ** filenames, unsigned int number_of_filenames) {
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

image * resize_images(image * images, unsigned int number_of_images) {
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

float * images_to_data(image * images, unsigned int number_of_images) {
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

void free_images(image * images, unsigned int number_of_images) {
	for (int n=0; n<number_of_images; n++) {
		free_image(images[n]);                    // image.c
	}
}

// --------------- Detect on the Image ---------------

// Detect on Image: this function uses other functions not from this file
detection ** run_detector(float * data, unsigned int number_of_images) 
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
	float *X = data+image_size*processing_index;
        time = clock();
        //network_predict(net, X);
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

	for (int b=0; b<batch_size; b++) {
		float hier_thresh = 0.5;
		int ext_output = 1, letterbox = 0, nboxes = 0;
		detection *dets = get_network_boxes(&net, net.w, net.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox, b );
		if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
		//draw_detections_v3(resized_images[i], dets, nboxes, thresh, names, l.classes, ext_output);

		//char buffer[256];
		//sprintf(buffer,"predictions_%d",i);
		//save_image_png(resized_images[i], buffer);    // image.c

		image_dets[processing_index+b]=dets;

		//free_image(resized_images[i]);                    // image.c
		//free_detections(dets, nboxes);
	}

	processing_index+=batch_size;
    }
    return image_dets;
}



// --------------- Detect on the Video ---------------

// get command line parameters and load objects names
void init_detector(int argc, char **argv)
{
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
    while (fgets(buffer, 255, (FILE*)fp)) {
        names[obj_count] = calloc(strlen(buffer)+1, sizeof(char));
        strcpy(names[obj_count], buffer);
        names[obj_count][strlen(buffer) - 1] = '\0'; //remove newline
        ++obj_count;
    }
    fclose(fp);
    int classes = obj_count;

    batch_size=2;
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
}


void close_detector() {
   //TODO 

    int i;
    for (i = 0; i < obj_count; ++i) free(names[i]);
    free(names);
}


