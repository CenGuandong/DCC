#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

//#define Lambda 3

using namespace std;

namespace caffe {
  template <typename Dtype>
  void JaccardLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
  {
    	Layer<Dtype>::LayerSetUp(bottom,top);
    	this->layer_param_.add_loss_weight(Dtype(1));
  }

  template <typename Dtype>
  void JaccardLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
  {
  	LossLayer<Dtype>::Reshape(bottom, top);
        vector<int> loss_shape(0);
    	top[0]->Reshape(loss_shape);
  }

  template <typename Dtype>
  void JaccardLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) 
  {
        int num = bottom[0]->num();
        const Dtype* input_data = bottom[0]->cpu_data();
  	const Dtype* target = bottom[1]->cpu_data();
  	Dtype Lambda_ = this->layer_param_.jaccard_loss_param().w_();
  	Dtype count_fg = 0;
  	Dtype count_bg = 0;
  	Dtype count_bg_2 = 0;
  	Dtype sum_fg = 0;
  	Dtype sum_bg = 0;
  	Dtype sum_bg_2 = 0;
  	Dtype jaccardloss = 0;
  	int dim = bottom[0]->count() / bottom[0]->num();
	for (int i = 0; i < num; ++i) 
	{
		count_fg = 0;
		count_bg = 0;
		count_bg_2 = 0;
		sum_fg = 0;
  		sum_bg = 0;
  		sum_bg_2 = 0;
		for (int j = 0; j < dim; j ++) 
		{
			Dtype labelValue = target[i*dim+j]*0.00392157;
			if (labelValue > 0.9) {
	        		count_fg ++;
	        		sum_fg += input_data[i*dim + j];
	    		}
	    		else if(labelValue < 0.05)
	    		{	
	    			count_bg ++;	
	    			sum_bg += input_data[i*dim + j];
	    		}
			else
			{
				count_bg_2 ++;
				sum_bg_2 += input_data[i*dim + j];
			}
	    	}
	    	jaccardloss += 1 - sum_fg / (count_fg + sum_bg + Lambda_ * sum_bg_2);
	 }
	 top[0]->mutable_cpu_data()[0] = jaccardloss / num;
  }


  template <typename Dtype>
  void JaccardLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
  {
  	if (propagate_down[1]) {
    	LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  	}
  	if (propagate_down[0]) {
	    // First, compute the diff
	    const int count = bottom[0]->count();
	    const int num = bottom[0]->num();
	    const Dtype* input_data = bottom[0]->cpu_data();
	    const Dtype* target = bottom[1]->cpu_data();
	    Dtype* target_scale = bottom[1]->mutable_cpu_data();
	    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();   
	    caffe_cpu_scale(count, Dtype(0.00392157), target,target_scale);
	    Dtype Lambda_ = this->layer_param_.jaccard_loss_param().w_();
	    Dtype count_fg = 0;
	    Dtype count_bg = 0;
	    Dtype count_bg_2 = 0;
	    Dtype sum_fg = 0;
  	    Dtype sum_bg = 0;
  	    Dtype sum_bg_2 = 0;
	    int dim = bottom[0]->count() / bottom[0]->num();
	    for (int i = 0; i < num; ++i) 
	    {
	    	count_fg = 0;
	    	count_bg = 0;
	    	count_bg_2 = 0;
	    	sum_fg = 0;
  		sum_bg = 0;
  		sum_bg_2 = 0;
	    	for (int j = 0; j < dim; j ++) {
			Dtype labelValue = target[i*dim+j];
		   	if (labelValue > 0.9)   //text region
			{ 
		        	count_fg ++;
		        	sum_fg += input_data[i*dim + j];
			}
			else if(labelValue < 0.05)   //general background 
			{
		        	count_bg ++;
		        	sum_bg += input_data[i*dim + j];
			}
			else   //weighted background
		    	{
			    	count_bg_2 ++; 
			    	sum_bg_2 += input_data[i*dim + j];
		    	}
	     	}
	    	for (int j = 0; j < dim; j ++) 
		{
	        	if (target[i*dim+j] > 0.9) {
	                	bottom_diff[i * dim + j] = -1 / (count_fg + sum_bg + Lambda_ * sum_bg_2);
	        	}
	        	else if(target[i*dim+j] < 0.05)
	        	{
	        		bottom_diff[i * dim + j] = sum_fg / ((count_fg + sum_bg + Lambda_ * sum_bg_2) * (count_fg + sum_bg + Lambda_ * sum_bg_2));
	        	}
	        	else
	        	{
	        		bottom_diff[i * dim + j] = Lambda_ * sum_fg / ((count_fg + sum_bg + Lambda_ * sum_bg_2) * (count_fg + sum_bg + Lambda_ * sum_bg_2));
	        	}
	     	}
	    }
	    const Dtype loss_weight = top [0]->cpu_diff()[0];
	    caffe_scal(count, loss_weight / num, bottom_diff);
	}
}

#ifdef CPU_ONLY
STUB_GPU(JaccardLossLayer);
#endif

INSTANTIATE_CLASS(JaccardLossLayer);
REGISTER_LAYER_CLASS(JaccardLoss);

}  // namespace caffe
