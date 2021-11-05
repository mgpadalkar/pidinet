/*
 * test_crack_detector.cpp
 *
 * Created on: July 9, 2019
 * Author: milind
 *
 * An app to test the class to perform crack detector using
 * Python code and Tensorflow trained models.
 *
 */


#include <edge_detector.h>


int main(int argc, char* argv[])
{

  // presets
  std::string image_path = "test_image.bmp";
  std::string light_name = "light_0";
  std::string model_root = "trained_models";

  // check arguments
  if(argc > 1)
    image_path = argv[1];

  if(argc > 2)
    light_name = argv[2];

  if(argc > 3)
    model_root = argv[3];


  // print image and model paths
  std::cout << "image path: " << image_path << std::endl;
  std::cout << "light name: " << light_name << std::endl;
  std::cout << "model root: " << model_root << std::endl;


  // load image
  cv::Mat image = cv::imread( image_path, cv::IMREAD_COLOR );
  if(image.empty())
  {
    std::cout << "Unable to load image\n";
    return(-1);
  }


  // // initialize tile mask
  // cv::Mat tile_mask = cv::Mat(image.size(), CV_8UC1, cv::Scalar(255));

  // initialize cracks
  cv::Mat edge = cv::Mat::zeros(image.size(), CV_8UC1);

  // set parameters for crack detector
  float resize_factor 			= 0.3;

  // object for crack detector
  // CrackDetector cdetect;
  // EdgeDetector cdetect(resize_factor, patch_dimension, stride, infer_in_batch, probability_based_inference, prob_cutoff, tile_th);
  EdgeDetector cdetect;
  int output_code = cdetect.detect_edges(image, edge);

  // show input
  cv::namedWindow("input image", cv::WINDOW_NORMAL);
  cv::imshow("input image", image);
  if (!edge.empty())
  {
    // show edges
    cv::namedWindow("detected edges", cv::WINDOW_NORMAL);
    cv::imshow("detected edges", edge);
  }
  cv::waitKey(0);


  // return
  return(0);
}
