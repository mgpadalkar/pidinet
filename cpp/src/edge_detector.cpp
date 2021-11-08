/*
 * edge_detector.cpp
 *
 * Created on: July 9, 2019
 * Author: milind
 *
 * Updated on: July 23, 2020
 *
 * Source for the class to perform crack detector using
 * Python code and Tensorflow trained models.
 *
 */


#include <edge_detector.h>


// constructor
EdgeDetector::EdgeDetector()
{
  m_module_name			= "edge_detector";
  m_function_name		= "detect_edges";
  m_load_name   		= "load_model";

  m_model_arg                   = "pidinet_tiny_converted";
  m_config_arg                  = "carv4";
  m_checkpoint_arg              = "trained_models/table5_pidinet-tiny-l.pth";
  m_sa_arg                      = false;
  m_dil_arg                     = false;
  m_resize_factor_arg		= 1.0;

  m_module                      = NULL;
  m_function                    = NULL;
  m_model                       = NULL;
  m_device                      = NULL;

  m_argc 			= 1;
  m_argv[0]			= strdup("edge_detector");

  // initialize
  init_detector();
}


// parameterized constructor
EdgeDetector::EdgeDetector
(
  std::string model_arg, std::string config_arg, std::string checkpoint_arg,
  bool sa_arg, bool dil_arg, float resize_factor,
  std::string module_name, std::string function_name, std::string load_name
)
{
  m_module_name			= module_name;
  m_function_name		= function_name;
  m_load_name   		= load_name;

  m_model_arg                   = model_arg;
  m_config_arg                  = config_arg;
  m_checkpoint_arg              = checkpoint_arg;
  m_sa_arg                      = sa_arg;
  m_dil_arg                     = dil_arg;
  m_resize_factor_arg		= resize_factor;

  m_module                      = NULL;
  m_function                    = NULL;
  m_model                       = NULL;
  m_device                      = NULL;

  m_argc 			= 1;
  m_argv[0]			= strdup("edge_detector");

  // initialize
  init_detector();
}


void EdgeDetector::init_detector()
{
  // init module
  bool module_ok = init_module(m_module_name, m_module);

  // init function
  bool function_ok = init_function(m_module, m_function_name, m_function);
  
  // load model and device
  bool load_ok = load_model_device(m_module, m_load_name, m_model, m_device);
}


// destructor
EdgeDetector::~EdgeDetector()
{
  // decrement the object references
  if (m_module != NULL)
    Py_XDECREF(m_module);

   if (m_function != NULL)
    Py_XDECREF(m_function);
   
   if (m_model != NULL)
    Py_XDECREF(m_model);

   if (m_device != NULL)
    Py_XDECREF(m_device);
}


// initialize a module
bool EdgeDetector::init_module(std::string module_name, PyObject *&module)
{
  // initialize Python embedding
  Py_Initialize();

  // set the command line arguments (can be crucial for some python-packages, like tensorflow)
  #ifdef PYTHON2
  PySys_SetArgv(m_argc, m_argv);
  #else
  wchar_t ** w_argv = new wchar_t*;
  *w_argv = Py_DecodeLocale(m_argv[0], NULL);
  PySys_SetArgv(m_argc, w_argv);
  #endif

  // add the current folder to the Python's PATH
  PyObject *sys_path = PySys_GetObject(strdup("path"));
  PyList_Append(sys_path, PyUnicode_FromString("./pyscripts"));


  // // print python version
  // PyRun_SimpleString("import sys\n"
  //                    "print(sys.version)\n");


  // this macro is defined be NumPy and must be included
  import_array1(-1);


  // load our python script
  module = PyImport_ImportModule(module_name.c_str());

  // message
  bool module_ok = module != NULL;
  if (module_ok)
  {
    std::cout << "=> Python module \"" << module_name << "\" initialized!\n";
  }
  else
  {
    std::cout << "=> Python module \"" << module_name << "\" NOT initialized!\n";
  }

  // return
  return module_ok;
}

// initialize a function
bool EdgeDetector::init_function
(
  PyObject *module,
  std::string function_name,
  PyObject *&function
)
{
  if (module == NULL)
  {
    return false;
  }

  // get dictionary of available items in the module
  PyObject *dict = PyModule_GetDict(module);

  // grab the functions we are interested in
  function = PyDict_GetItemString(dict, function_name.c_str());

  // message
  bool function_ok = function != NULL;
  if (function_ok)
  {
    std::cout << "=> Function \"" << function_name << "\" initialized!\n";
  }
  else
  {
    std::cout << "=> Function \"" << function_name << "\" NOT initialized!\n";
  }

  // return
  return function_ok;
}


bool EdgeDetector::load_model_device
(
  PyObject *module,
  std::string function_name,
  PyObject *&model,
  PyObject *&device
)
{
  // initialize function
  PyObject *function;
  bool function_ok = init_function(module, function_name, function);
  if (!function_ok)
  {
    return false;
  }

  // preprare args for calling function
  PyObject* args = Py_BuildValue("({s:s, s:s, s:s, s:O, s:O, s:d})",
    "model", m_model_arg.c_str(),
    "config", m_config_arg.c_str(),
    "checkpoint", m_checkpoint_arg.c_str(),
    "sa", m_sa_arg ? Py_True : Py_False,
    "dil", m_dil_arg ? Py_True : Py_False,
    "resize_factor", m_resize_factor_arg);

  // validate args
  if (args == NULL)
  {
   std::cout << "=> Problem loading args before calling \"" << function_name << "\"\n";
    return false;
  }


  // execute the function
  PyObject* py_res = PyEval_CallObject(function, args);
  if (py_res == NULL)
  {
    std::cout << "=> Problem executing function \"" << function_name << "\"\n";
    return false;
  }


  // get results
  PyObject *py_model, *py_device;
  if(PyArg_ParseTuple(py_res, "OO", &py_model, &py_device))
  {
    std::cout << "=> Function \"" << function_name << "\" called successfully!\n";
    model = py_model;
    device = py_device;
  }
  else
  {
    std::cout << "=> Problem getting output from function \"" << function_name << "\"!\n";
    return false;
  }

  // message
  bool model_ok = model != NULL;
  bool device_ok = device != NULL;
  if (model_ok)
  {
    std::cout << "=> Model \"" << m_model_arg << "\" loaded successfully!\n";
  }
  else
  {
    std::cout << "=> Model \"" << m_model_arg << "\" NOT loaded!\n";
    return (model_ok);
  }
  if (!device_ok)
  {
    std::cout << "=> Problem loading device in function  \"" << function_name << "\"!\n";
    return (device_ok);
  }
  return true;
}


// public call
int EdgeDetector::detect_edges
(
  cv::Mat image,
  cv::Mat &edge
)
{
  //init output code
  int output_code = -1;

  // check for resize
  if (m_resize_factor_arg != 1.)
  {
    // resize for processing
    cv::Size image_size = image.size();
    cv::resize(image, image, cv::Size(0, 0), m_resize_factor_arg, m_resize_factor_arg);

    // call python wrapper
    output_code = detect_edges_wrapper(image, edge);

    // resize back
    cv::resize(edge, edge, image_size, 0, 0, cv::INTER_NEAREST);
  }
  else
  {
    // call python wrapper
    output_code = detect_edges_wrapper(image, edge);
  }

  // return
  return output_code;
}


// real function
int EdgeDetector::detect_edges_wrapper
(
  cv::Mat image,
  cv::Mat &edge
)
{
  // check if function exists
  if (m_function == NULL)
    return -1;
  
  // initialize output
  edge.setTo(0);

  // // convert input image to 1 channel
  // if(image.channels() > 1)
  //   cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);

  // convert the cv::mat to numpy.array
  PyObject *py_image;
  // cvMat_1Ch_to_Numpy(image, py_image);
  cvMat_to_Numpy(image, py_image);


  // create a Python-tuple of arguments for the function call
  // "()" means "tuple", "O" means "object", "i" mean "integer", "d" means "double", "s" means "string"
  // example: PyObject* args = Py_BuildValue("(OOi)", mat, mat, 123);
  // see detailed explanation here: https://docs.python.org/2.0/ext/buildValue.html
  PyObject* args = Py_BuildValue("(OOOf)",
    py_image, m_model, m_device, 1.0);

  // check if anything is null
  if(py_image == NULL)
  {
    std::cout << "=> Image is null\n";
    return(-2);
  }
  if(args == NULL)
  {
    std::cout << "=> args for calling function \"" << m_function_name << "\" are  null\n";
    return(-3);
  }


  // execute the function
  PyObject* py_res = PyEval_CallObject(m_function, args);
  if (py_res == NULL)
  {
    std::cout << "=> Problem executing function \"" << m_function_name << "\"!\n";
    return(-4);
  }

  PyObject* py_edge;
  int dummy;
  if(PyArg_Parse(py_res, "O", &py_edge))
  {
    std::cout << "=> Function \"" << m_function_name << "\" called successfully!\n";
  }
  else
  {
    std::cout << "=> Problem calling function \"" << m_function_name << "\"!\n";
    return(-5);
  }

  // convert result
  // Numpy_to_1Ch_cvMat(py_edge, edge);
  Numpy_to_cvMat(py_edge, edge);


  // decrement the object references
  Py_XDECREF(py_edge);
  Py_XDECREF(py_image);
  Py_XDECREF(args);

  return 0;
}

// function to convert cv::Mat to numpy array
void EdgeDetector::cvMat_to_Numpy(cv::Mat cv_mat, PyObject *&py_mat)
{
  // return if empty
  if(cv_mat.empty())
  {
    std::cout << "=> Received empty cv_Mat in \"cvMat_to_Numpy()\"\n";
    return;
  }

  // total number of elements
  int nElem = cv_mat.rows * cv_mat.cols * cv_mat.channels();
  
  // create an array of apropriate datatype
  uchar *data = new uchar[nElem];

  // copy the data from the cv::Mat object into the array
  std::memcpy(data, cv_mat.data, nElem * sizeof(uchar));

  // the dimensions of the matrix
  npy_intp shape[] = { cv_mat.rows, cv_mat.cols, cv_mat.channels() };
  int ndims = sizeof(shape)/sizeof(*shape);

  // convert the cv::Mat to numpy.array
  py_mat = PyArray_SimpleNewFromData(ndims, shape, NPY_UINT8, (void*) data);
}


// function to convert a numpy array to a cv::Mat array
// reference: https://stackoverflow.com/questions/39201533/how-can-convert-pyobject-variable-to-mat-in-c-opencv-code
void EdgeDetector::Numpy_to_cvMat(PyObject *py_mat, cv::Mat &cv_mat)
{

  // return if NULL
  if (py_mat == NULL)
  {
    std::cout << "=> Received empty py_mat in \"Numpy_to_cvMat\"\n";
    return;
  }

  // get size
  npy_intp *shape = PyArray_DIMS(py_mat);
  int ndims = PyArray_NDIM(py_mat);
  int rows = (int)shape[0];
  int cols = (int)shape[1];
  int channels = ndims > 2 ? (int)shape[2] : 1;
  if (channels != cv_mat.channels())
  {
    std::cout << "=> Warning: result is supposed to have " << cv_mat.channels() << 
      " channels but got " << channels << " channels.\n" << 
      "=> Will use " << channels << " channels.\n";
  }


  // create a cv::Mat
  void *data = PyArray_DATA(py_mat);
  int type = CV_MAKETYPE(CV_8U, (channels));
  cv::Mat mat(rows, cols, type, data);
  
  // copy to our mat
  mat.copyTo(cv_mat);
}

