#ifdef __cplusplus
extern "C" {
#endif
void dn_init_detector(int argc, char **argv);
void dn_close_detector();
detection ** dn_run_detector(float * data, unsigned int number_of_images); 
float * dn_images_to_data(image * images, unsigned int number_of_images);
image * dn_resize_images(image * images, unsigned int number_of_images);
image * dn_load_images(const char ** filenames, unsigned int number_of_filenames);
void dn_free_images(image * images, unsigned int number_of_images);
char ** dn_get_names();
int dn_get_nboxes();
void dn_save_image(image im, char * filename);
void dn_draw_detections(image im,detection * dets);
void dn_free_detections(detection ** image_dets, unsigned int number_of_images);
#ifdef __cplusplus
}
#endif
