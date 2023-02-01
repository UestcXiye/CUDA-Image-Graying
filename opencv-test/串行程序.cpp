#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <math.h>
#include <time.h>
#include <io.h>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace cv;
using namespace std;

#define THREAD_NUM 256

//串行转换灰度图像
void rgb2grayincpu(unsigned char* const d_in, unsigned char* const d_out, uint imgheight, uint imgwidth) {
	//使用两重循环嵌套实现x方向和y方向的变换
	for (int i = 0; i < imgheight; i++) {
		for (int j = 0; j < imgwidth; j++) {
			d_out[i * imgwidth + j] = 0.299f * d_in[(i * imgwidth + j) * 3]
				+ 0.587f * d_in[(i * imgwidth + j) * 3 + 1]
				+ 0.114f * d_in[(i * imgwidth + j) * 3 + 2];
		}
	}
}


int Initfunc(string inputfilename, double& cpusumtime) {

	/*图片数据预处理*/
	//传入图片
	Mat srcImg = imread(inputfilename);
	FILE* fp;//创建运行时间文件

	//读取图片像素值
	int imgHeight = srcImg.rows;
	int imgWidth = srcImg.cols;

	Mat grayImg(imgHeight, imgWidth, CV_8UC1, Scalar(0));    //输出灰度图
	int hist[256];    //灰度直方图统计数组
	memset(hist, 0, 256 * sizeof(int));    //对灰度直方图数组初始化为0


	/*CPU串行开始*/
	//串行灰度化
	//计时开始
	auto cpustart = chrono::system_clock::now();
	//调用主函数
	rgb2grayincpu(srcImg.data, grayImg.data, imgHeight, imgWidth);

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	//计时结束
	auto cpuend = chrono::system_clock::now();
	//计算时间差
	auto cpuduration = chrono::duration_cast<chrono::microseconds>(cpuend - cpustart);
	double cput = cpuduration.count();
	//微秒转化为秒
	double cputime = cput / 1000000;
	cpusumtime += cputime;
	//打印串行执行时间
	cout << setiosflags(ios::fixed) << setprecision(10) << "cpu  exec time： " << cputime << " s" << endl;
	//printf("cpu  exec time is %.10lg s\n", cputime / 1000000);

	/*输出灰度图片*/
	try {
		int len = inputfilename.length();
		cout << "inputfilename.length:" << len << endl;
		string str = "./GrayPicture/";
		imwrite(str + inputfilename.substr(10, len - 14) + "_to_gray.png", grayImg, compression_params);
		cout << str + inputfilename.substr(10, len - 14) + "_to_gray.png" << endl;

		//在GrayPicture文件夹中，生成灰度变换后的结果图片
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "图像转换成PNG格式发生错误：%s\n", ex.what());
		return 1;
	}
	return 0;
}

//批量读取图片
void getFiles(string path, vector<string>& files) {
	//文件句柄
	intptr_t hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			//如果是目录,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib & _A_SUBDIR)) {
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else {
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


int main() {
	//图片文件路径，在项目文件下的Picture文件夹里面
	string filePath = "./Picture";
	vector<string> files;
	//读取图片文件
	getFiles(filePath, files);
	//读取图片数量
	int size = files.size();
	//输出图片数量
	cout << "图片数量：" << size << endl;

	double cpusumtime = 0;
	for (int i = 0; i < size; i++) {
		cout << "第 " << i + 1 << "/" << size << " 张图片" << endl;
		cout << files[i].c_str() << endl;
		Initfunc(files[i].c_str(), cpusumtime);
		cout << endl;
	}

	cout << "cpusumtime：" << cpusumtime << " s" << endl;

	return 0;
}
