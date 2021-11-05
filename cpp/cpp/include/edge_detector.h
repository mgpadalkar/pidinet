/*
 * edge_detector.hpp
 *
 * Created on: July 9, 2019
 * Author: milind
 *
 * A class to perform crack detector using
 * Python code and Tensorflow trained models.
 *
 */

#ifndef EDGE_DETECTOR_HPP
#define EDGE_DETECTOR_HPP


// python include
#include <Python.h>

// python packages
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


// OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

// other required headers
#include <iostream>
#include <string>


// class to handle the reference data
class EdgeDetector
{

  private:
    
    std::string m_module_name;
    std::string m_function_name;
    std::string m_load_name;

    std::string m_model_arg;
    std::string m_config_arg;
    std::string m_checkpoint_arg;
    bool m_sa_arg;
    bool m_dil_arg;
    float m_resize_factor_arg;

    int m_argc;
    char* m_argv[1];

    PyObject *m_module;
    PyObject *m_function;
    PyObject *m_model;
    PyObject *m_device;

    void init_detector();
    bool init_module(std::string module_name, PyObject *&module);
    bool init_function(PyObject *module, std::string function_name, PyObject *&function);
    bool load_model_device(PyObject *module, std::string function_name, PyObject *&model, PyObject *&device);

    void cvMat_to_Numpy(cv::Mat cv_mat, PyObject *&py_mat);
    void Numpy_to_cvMat(PyObject *py_mat, cv::Mat &cv_mat);
    int detect_edges_wrapper(cv::Mat image, cv::Mat &edge_mask);

  public:
    EdgeDetector();
    ~EdgeDetector();

    // int detect_edges(std::string light_name, std::string model_root, cv::Mat image, cv::Mat tile_mask, cv::Mat &edge_mask);
    int detect_edges(cv::Mat image, cv::Mat &edge_mask);
};



#endif /* EDGE_DETECTOR_HPP */
