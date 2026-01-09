This is an experiment based on ORL database, uses K-L transformation and PCA methods, feature faces are obtained, and the input face is recognized by outputting the most similar feature face. 
基于ORL数据库，利用K-L变换以及PCA方法，求得特征脸，并对输入的人脸进行识别，输出特征最相似的人脸。
ORL_main.cpp:
The cpp code of the experiment, use g++ to compile it or just use the given .exe files.
实验C++代码，使用g++编译或直接使用给出的.exe文件
ORL_main.exe:
Compiled from ORL_main.cpp, the value of K (number of Eigenfaces) is set to be 50.
由ORL_main.cpp编译出来，K值（特征脸数量）设置为50
ORLtest.exe:
Compiled from ORL_main.cpp, the value of K (number of Eigenfaces) is set to be 40.
由ORL_main.cpp编译得到，K值（特征脸数量）设置为40
stb_image.h/stb_image_write.h
cr:
用于读取png files
