How to run the program:

We have given testing dataset in Dataset folder for DAVIS and FBMS.

1. Now run the main.py file
2. It will ask for enter your choice where 
	1.Is for running the algorithm on your video
	2.Is for running the algorithm on your dataset
3. If you choose one 
	You have to provide the path of the video tutorial
4. if you choose two 
	You have to provide the path of the dataset './Dataset/video/'. 
	Don't put the path of single video from output because it will takes all the video from 
	the given path and make the object of the names of video so the program runs for loop on that array object.
5. Please, use the provided weights of cnn model in repository.
6. Now, wait till the processing done. After completion you will see two folders i.e. static_saliency_output' and 'video_saliency_output'.
7. We have to use static_saliency_output result in cnn model of dynamic video salient object detection that's why we also stored the output of static salient video.



