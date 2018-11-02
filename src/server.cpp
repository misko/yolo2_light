#include <stdio.h>
#include <stdlib.h>

#include "box.h"
#include "additionally.h"
#include "server_functions.h"

#include <string>
#include <vector>
#include <fstream>

int main(int argc, char **argv)
{
    int i;
    for (i = 0; i < argc; ++i) {
        if (!argv[i]) continue;
        strip(argv[i]);
    }

    if (argc < 2) {
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }

    //load up the detector
    dn_init_detector(argc, argv);

    //read in the input list
    char *input_list = find_char_arg(argc, argv, "-list", "wtf");
    std::string str;
    std::vector<std::string> lines;
    std::ifstream in(input_list);
    while (std::getline(in, str)) {
	if(str.size() > 0) {
		lines.push_back(str);
	}
    }
    unsigned int number_of_filenames=lines.size();
    const char * filenames[number_of_filenames];
    for (int i=0; i<number_of_filenames; i++) {
	filenames[i]=lines[i].c_str();
    }

    //load file images and put them into a big matrix
    image * images = dn_load_images(filenames,number_of_filenames);
    image * resized_images = dn_resize_images(images,number_of_filenames);
    float * data = dn_images_to_data(resized_images,number_of_filenames);

    char ** names = dn_get_names(); 
    for (int x=0; x<3; x++) {
	int * nboxes = (int*)malloc(sizeof(int)*number_of_filenames);
	if (nboxes==NULL) {
		fprintf(stderr,"Failed to amlloc \n");
		exit(1);
	}
    	detection ** image_dets = dn_run_detector(data,number_of_filenames,nboxes);
	for (int i=0; i<number_of_filenames; i++) {
		fprintf(stderr,"output for image %d\n",i);
		detection * dets = image_dets[i];
		for (int d=0; d<nboxes[i]; d++){
			for (int j=0; j<dets[d].classes; j++) {
				if (dets[d].prob[j]>0.25) {
					fprintf(stderr,"%s %0.2f\n",names[j],dets[d].prob[j]);
				}
			}
		}

	}
    }
    return 0;
}
