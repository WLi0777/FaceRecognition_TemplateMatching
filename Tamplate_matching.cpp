#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    Mat src;
    Mat image_gray;
    VideoCapture capture;
    Mat templateImg = imread("Template.jpg"); // Template image
    string video_filename = "OriginalVideo.avi"; // Video to process
    
    Mat resultImg;
    int resultImg_cols, resultImg_rows;
    
    capture.open(video_filename.c_str());
    
    int rate = capture.get(CAP_PROP_FPS);
    cout << rate << endl;
    
    VideoWriter writer("ProcessedVideo.avi", writer.fourcc('M', 'J', 'P', 'G'), rate, Size(960, 544));
    
    if (!capture.isOpened()) {
        printf("--(!)Error opening video capture\n");
        return -1;
    }
    
    namedWindow("src", WINDOW_AUTOSIZE);
    char c = 0;
    
    CascadeClassifier c1;
    
    bool res = c1.load("haarcascade_frontalface_alt.xml");
    
    if (res == true) {
        cout << "xml ok" << endl;
    }
    else {
        cout << "xml error" << endl;
        return -1;
    }
    
    Mat dst;
    Mat dst1;
    Mat imagel;
    vector<Rect> faces;
    
    Mat image;
    Mat tamp2;
    
    while (capture.read(src)) {
        dst = Mat::zeros(src.size(), src.type());
        cvtColor(src, dst, COLOR_BGR2GRAY);
        dst1 = Mat::zeros(dst.size(), dst.type());
        equalizeHist(dst, dst1);
        c1.detectMultiScale(dst1, faces, 1.1, 3, 0, Size(50, 50));
        
        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(src, faces[i], Scalar(255, 0, 255), 2, LINE_8, 0);
            image = src(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height)); // Save
            resize(image, imagel, Size(100, 100));
            
            // imwrite(format("C:\\Users\\Desktop\\p\\rrr%d.jpg",count),imagel);
            
            Mat showImg = imagel.clone();
            resultImg_cols = abs(imagel.cols - templateImg.cols + 1);
            resultImg_rows = abs(imagel.rows - templateImg.rows + 1);
            resultImg.create(resultImg_cols, resultImg_rows, CV_32FC1);
            
            cvtColor(imagel, image_gray, COLOR_BGR2GRAY);
            equalizeHist(image_gray, image_gray);
            
            matchTemplate(imagel, templateImg, resultImg, TM_CCOEFF);
            normalize(resultImg, resultImg, 0, 1, NORM_MINMAX);
            
            double minValue, maxValue;
            Point minLoc, maxLoc;
            Point matchLoc;
            
            minMaxLoc(resultImg, &minValue, &maxValue, &minLoc, &maxLoc);
            
            if (faces.size() == 1) {
                if (maxValue >= 0.7) {
                    putText(src, "Name", Point(faces[i].x, faces[i].y), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 255), 1, LINE_8);
                }
            }
            else {
                if (maxValue == 1) {
                    putText(src, "Name", Point(faces[i].x, faces[i].y), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 255), 1, LINE_8);
                }
                else {
                    putText(src, "Unkown", Point(faces[i].x, faces[i].y), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 255), 1, LINE_8);
                }
            }
            
        }
        
        writer<<src;
        
        imshow("src",src);
        c = waitKey(30);
        if(c == 27)
        {
            break;
        }
    }
    waitKey(30);
    return 0;
    
}
