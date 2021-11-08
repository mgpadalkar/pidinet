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
  std::string image_path;
  std::string model_name = "pidinet_tiny_converted";
  std::string config     = "carv4";
  std::string checkpoint = "trained_models/table5_pidinet-tiny-l.pth";
  bool sa                = false;
  bool dil               = false;
  float resize_factor    = 1.0;


  // check arguments
  if(argc > 1)
    image_path = argv[1];
  else
  {
    std::cout << "syntax: " << argv[0] <<
    "<model_name> <config> <checkpoint> <sa> <dil> <resize_factor>\n";
    return(-1);
  }

  if(argc > 2)
    model_name = argv[2];

  if(argc > 3)
    config = argv[3];

  if(argc > 4)
    checkpoint = argv[4];

  if(argc > 5)
    sa = std::atoi(argv[5]) > 0;

  if(argc > 6)
    dil = std::atoi(argv[6]) > 0;

  if(argc > 7)
    resize_factor = std::atof(argv[7]);


  // print image and model paths
  std::cout << "model_name: " << model_name << std::endl;
  std::cout << "config: " << config << std::endl;
  std::cout << "checkpoint: " << checkpoint << std::endl;
  std::cout << "sa: " << sa << std::endl;
  std::cout << "dil: " << dil << std::endl;
  std::cout << "resize_factor: " << resize_factor << std::endl;


  // load image
  cv::Mat image = cv::imread( image_path, cv::IMREAD_COLOR );
  if(image.empty())
  {
    std::cout << "Unable to load image\n";
    return(-1);
  }

  // initialize output
  cv::Mat edge = cv::Mat::zeros(image.size(), CV_8UC1);

  // EdgeDetector cdetect; // default
  EdgeDetector cdetect(model_name, config, checkpoint, sa, dil, resize_factor);
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
