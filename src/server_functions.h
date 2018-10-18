#ifdef __cplusplus
extern "C" {
#endif
void init_detector(int argc, char **argv);
void close_detector();
detection ** run_detector(float * data, unsigned int number_of_images); 
float * images_to_data(image * images, unsigned int number_of_images);
image * resize_images(image * images, unsigned int number_of_images);
image * load_images(const char ** filenames, unsigned int number_of_filenames);
void free_images(image * images, unsigned int number_of_images);
char ** get_names();
int get_nboxes();
#ifdef __cplusplus
}
#endif
