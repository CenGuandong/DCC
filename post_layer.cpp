#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using namespace cv;
using namespace std;

namespace caffe {


float calDistance(float w, float h, float mValue)
{
  float mean_w = 258.0;
  float mean_h = 35.0;
  return exp(-1*((std::abs(w - mean_w) / w + 1*std::abs(h - mean_h) / h) / (2*mValue)));
}

void HorizonProjection(const Mat& src, Mat& dst)
{
  CV_Assert(src.depth() != sizeof(uchar));
  dst.create(src.rows, 1, CV_32F);
  int i, j;
  const uchar* p;
  float* p_dst;
  for (i = 0; i < src.rows; i++){
    p = src.ptr<uchar>(i);
    p_dst = dst.ptr<float>(i);
    p_dst[0] = 0;
    for (j = 0; j < src.cols; j++){
      p_dst[0] += p[j];
    }
  }
}

void HorizontalCutting(const Mat & src, const Mat & srcOri, Mat & dst, int &diffTop, int &diffDown)
{
  Mat img = src.clone();
  Mat hp;
  HorizonProjection(img, hp);
  Point maxL;
  cv::minMaxLoc(hp, NULL, NULL, NULL, &maxL);
  int mid = maxL.y + 1;
  vector<int> top, down;
  top.clear();
  down.clear();
  for (int i = 0; i < mid; i++)
    top.push_back((int)hp.at<float>(i, 0));
  for (int i = mid - 1; i < hp.rows; i++)
    down.push_back((int)hp.at<float>(i, 0));
  vector<int>::iterator it;
  int maxValue = 0;
  int topY = 0, downY = 0;
  for (it = top.begin(); it != top.end() - 1; it++)
  {
    int currentValue = abs(*(it + 1) - *it);
    if (currentValue >= maxValue)
    {
      int i = it - top.begin();
      topY = i;
      maxValue = currentValue;
    }
  }
  topY = topY - 1 < 1 ? 1 : (topY - 1);
  maxValue = 0;
  for (it = down.begin(); it != down.end() - 1; it++)
  {
    int currentValue = abs(*(it + 1) - *it);
    if (currentValue >= maxValue)
    {
      int i = it - down.begin();
      downY = i;
      maxValue = currentValue;
    }
  }
  downY += mid;
  dst = srcOri.rowRange(topY - 1, std::min(downY + 8,img.rows));
  diffTop = topY;
  diffDown = srcOri.rows - downY;
}

void VerticalProjection(const Mat& src, Mat& dst)
{
  CV_Assert(src.depth() != sizeof(uchar));
  dst.create(1, src.cols, CV_32F);
  int i, j;
  const uchar* p;
  float* p_dst = dst.ptr<float>(0);
  for (j = 0; j < src.cols; j++){
    p_dst[j] = 0;
    for (i = 0; i < src.rows; i++){
      p = src.ptr<uchar>(i);
      p_dst[j] += p[j];
    }
  }
}

void VerticalCutting(const Mat & src, const Mat & srcOri, Mat & dst, int &diffLeft, int &diffRight)
{
  Mat img = src.clone();;
  Mat vp;
  VerticalProjection(img, vp);
  Point maxL;
  cv::minMaxLoc(vp, NULL, NULL, NULL, &maxL);
  int mid = img.cols / 2;
  vector<int> left, right;
  left.clear();
  right.clear();
  for (int i = 0; i < mid; i++)
    left.push_back((int)vp.at<float>(0, i));
  for (int i = mid - 1; i < vp.cols; i++)
    right.push_back((int)vp.at<float>(0, i));
  vector<int>::iterator it;
  int maxValue = 0;
  int leftY = 0, rightY = 0;
  for (it = left.begin(); it != left.end() - 1; it++)
  {
    int currentValue = abs(*(it + 1) - *it);
    if (currentValue >= maxValue)
    {
      int i = it - left.begin();
      leftY = i;
      maxValue = currentValue;
    }
  }
  maxValue = 0;
  for (it = right.begin(); it != right.end() - 1; it++)
  {
    int currentValue = abs(*(it + 1) - *it);
    if (currentValue >= maxValue)
    {
      int i = it - right.begin();
      rightY = i;
      maxValue = currentValue;
    }
  }
  rightY += mid;
  int offsetLeft = -0, offsetRight = 0;
  if (((rightY + offsetRight) >= srcOri.cols ? srcOri.cols : (rightY + offsetRight)) - ((leftY + offsetLeft) <= 1 ? 1 : (leftY + offsetLeft)) > 210)
    dst = srcOri.colRange((leftY + offsetLeft) <= 1 ? 1 : (leftY + offsetLeft), (rightY + offsetRight) >= srcOri.cols ? srcOri.cols : (rightY + offsetRight));
  else
    dst = srcOri;
  diffLeft = leftY;
  diffRight = srcOri.cols - rightY;
}


template <typename Dtype>
inline Mat postProcessing(Dtype* top_data,Dtype* original_data,int h,int w,float binary_threshold_,float area_threshold_,float mean_h_,float mean_w_,vector<int> offset)
{
  Mat re_original = Mat::ones(h,w,CV_8U);
  uchar* pxVec_original = re_original.ptr<uchar>(0);
  for(int i = 0;i < re_original.rows; i++)
  {
    pxVec_original = re_original.ptr<uchar>(i);
    for(int j = 0;j < re_original.cols;j++)
    {
      pxVec_original[j] = *(original_data + i*re_original.cols + j) * 1;
    }
  }
  Mat re = Mat::ones(h,w,CV_8U);
  uchar* pxVec = re.ptr<uchar>(0);
  for(int i = 0;i < re.rows; i++)
  {
    pxVec = re.ptr<uchar>(i);
    for(int j = 0;j < re.cols;j++)
    {
      pxVec[j] = *(top_data + i*re.cols + j) * 255;
    }
  }
  Mat re_bw;
  Mat display = re_original;
  Mat final;
  Mat dst = Mat::zeros(35,258,CV_8U);
  threshold(re,re_bw,0,255,CV_THRESH_OTSU);
  std::vector<std::vector<Point> > contours;
  std::vector<Vec4i> hierarchy;
  Mat mserSrc = re_bw.clone();
  findContours(mserSrc,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point(0,0));
  float maxConfidence = 0.0;
  for(int i=0;i < contours.size();i++)
  {
    double area = contourArea(contours[i]);
    RotatedRect rect = minAreaRect(contours[i]);
    Rect rectBouding = rect.boundingRect();
    if(area > h * w * area_threshold_)
    {
      float currentConfidence = calDistance((float)rectBouding.width, (float)rectBouding.height,(float)1.0);
      if (currentConfidence > 0.6)
      {
        float xLeftBottom = std::max(rectBouding.x + offset[0], 1);
        float yLeftBottom = std::max(rectBouding.y + offset[1], 1);
        float xRightTop = std::min(rectBouding.x + rectBouding.width + offset[2], w);
        float yRightTop = std::min(rectBouding.y + rectBouding.height + offset[3], h);
        display(cv::Rect(cv::Point(xLeftBottom, yLeftBottom), cv::Point(xRightTop, yRightTop))).copyTo(final);
        Mat gray_HC, gray_VC;
        re_bw(cv::Rect(cv::Point(xLeftBottom, yLeftBottom), cv::Point(xRightTop, yRightTop))).copyTo(gray_HC);
        int diffTop = 0, diffDown = 0, diffLeft = 0, diffRight = 0;
        Mat Rect_afterHC, finalRect;
        HorizontalCutting(gray_HC, final, Rect_afterHC, diffTop, diffDown);
        gray_VC = gray_HC.rowRange(diffTop, gray_HC.rows - diffDown);
        VerticalCutting(gray_VC, Rect_afterHC, finalRect, diffLeft, diffRight);
        finalRect = Rect_afterHC;
        float mValue = mean(gray_VC)[0]/255;
        float fDis = calDistance(rectBouding.width - (diffLeft + diffRight), rectBouding.height - (diffTop + diffDown), mValue);
        if (fDis > maxConfidence)
        {
          maxConfidence = fDis;
          dst = finalRect.clone();
        }
      }
    }
  }
  resize(dst,dst,Size(mean_w_,mean_h_));
  return dst;
}

template <typename Dtype>
void PostLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
  const PostParameter& post_param = this->layer_param_.post_param();
  binary_threshold_ = post_param.binary_threshold();
  area_threshold_ = post_param.area_threshold();
  mean_h_ = post_param.mean_h();
  mean_w_ = post_param.mean_w();
  offset.clear();
  switch (this->layer_param_.post_param().losstype()) {
  case PostParameter_Lt_SIGMOID:
      offset.push_back(-2);
      offset.push_back(-2); 
      offset.push_back(2);
      offset.push_back(2); 
    break;
  case PostParameter_Lt_JACCARD: 
      offset.push_back(0);  
      offset.push_back(0);
      offset.push_back(0);
      offset.push_back(0);
    break;
  default:
    LOG(FATAL) << "Unknown Loss Type.";
  }
  
}

template <typename Dtype>
void PostLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  top[0]->Reshape(bottom[0]->num(),3,(int)mean_h_,(int)mean_w_);
}

template <typename Dtype>
void PostLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_original = bottom[1]->cpu_data();
  const int count = bottom[0]->count();
  Dtype* bottom_0_data = new Dtype[count];
  Dtype* original_data = new Dtype[count];
  for (int i = 0; i < count; ++i) {
    bottom_0_data[i] = bottom_data[i];
  }
  for (int i = 0; i < count; ++i) {
    original_data[i] = bottom_original[i];
  }

  Mat dst = postProcessing(bottom_0_data,original_data,bottom[0]->shape(2),bottom[0]->shape(3),binary_threshold_,area_threshold_,mean_h_,mean_w_,offset);

  delete bottom_0_data;
  bottom_0_data = NULL;
  delete original_data;
  original_data = NULL;

  int out_width = int(mean_w_);
  int out_height = int(mean_h_);
  this->Reshape(bottom,top);
  Dtype* top_data = top[0]->mutable_cpu_data();

  for(int i = 0; i < dst.rows; i++)
  {
    uchar* dst_data = dst.ptr<uchar>(i);
    for(int j = 0; j < dst.cols; j++)
    {
      for(int c = 0;c < 3;c++)
      {
        top_data[out_width * out_height * c + i * out_width + j] = float(dst_data[j])/1; 
      }
    }
  }
}

template <typename Dtype>
void PostLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    //no backward
}

#ifdef CPU_ONLY
STUB_GPU(PostLayer);
#endif

INSTANTIATE_CLASS(PostLayer);
REGISTER_LAYER_CLASS(Post);


}  // namespace caffe
