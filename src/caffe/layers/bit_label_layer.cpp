#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BitLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = this->layer_param_.bit_label_param().num_output();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void BitLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  int num_output = this->layer_param_.bit_label_param().num_output();
  const Dtype* bottom_label = bottom[0]->cpu_data();
  Dtype* top_label = top[0]->mutable_cpu_data();

  for (int i = 0; i < bottom[0]->num(); ++i) {
    const int label_value = static_cast<int>(bottom_label[i]);
    int k = label_value;
    for (int j = num_output - 1; j >= 0 ; j--) {
      top_label[i * num_output + j] = Dtype(k % 2);
      if (k < 2) break;
      else k /= 2;
    }
  }
}

template <typename Dtype>
void BitLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(INFO) << "DO NOT BACK PROPAGATE!";
}

INSTANTIATE_CLASS(BitLabelLayer);
REGISTER_LAYER_CLASS(BitLabel);

}  // namespace caffe
