# Crack Detector (Local as well as Server)

It contains all the code for crack detection. The actual detection is performed in Python using Tensorflow. 

The training was performed using the follwing pipeline.

![Crack Detection Training Pipeline](crack_detector/docs/training_pipeline.png)

Below is a brief description of the contents.

**trained_models**: A folder containing the trained models.  

**inference_scripts**: Code to perform the actual crack detection using the trained models. It is equiped to run both locally as well as on a remote server.

**evaluation_scripts**: Code to evaluate the performance of crack detection at tile level. These are used for comparing the baseline architectures.

**include** and **src**: Source for a C++ CrackDetector class that internally passes a cv::Mat image to the Python code and receives a cv::Mat binary mask indicating the locations of the cracks.

**test**: A demo for using the C++ CrackDetector class to perform inference on a cv::Mat image. 

**make_and_run_scripts**: Code to compile the C++ CrackDetector as well as the demo app to generate an executable in the **bin** folder. It also includes a script to execute the demo.

## Crack Detection on a Remote Server

If the acquisition device has not enough computational resources for running tensorflow models, the crack detection can be executed on a more powerful remote machine.

Communication between the device and the server is implemented via a publisher/subscriber-like system based on [redis streams](https://redis.io/topics/streams-intro).

![Remote Crack Detection Pipeline](crack_detector/docs/remote_crack_detection.png)

For this to work, setup the following on the remote SERVER

1. Prepare the server by following instructions in [docs/crack_detector_server_preparation_instructions.txt](docs/crack_detector_server_preparation_instructions.txt)

2. Start redis server with docker engine

    open a new terminal window
    ```bash
    $ cd path/to/aen_acquisition_tool/crack_detector/inference_scripts
    $ docker-compose up
    ```

3. Start crack detector server

    open a new terminal window
    ```bash
    $ cd path/to/aen_acquisition_tool/crack_detector/inference_scripts
    $ chmod +x run_crack_detector_server.sh
    $ ./run_crack_detector_server.sh
    ```

Also, setup the following on the on the local DEVICE

4. Set

   ```LOCAL = False```

   in `path/to/aen_acquisition_tool/crack_detector/inference_scripts/crackDetector.py`  

5. Set redis' IP address

   ```REDIS_IP_ADDRESS = <server_ip_address>```

   in `path/to/aen_acquisition_tool/crack_detector/inference_scripts/crackDetector_remote.py`  


   
## Crack Detection on the Local Device

With this, crack detection can be performed locally on the device.

1.  Install the Pyhon and OpenCV library. See [docs/device_preparation_instructions_linux.md](docs/device_preparation_instructions_linux.md) for Linux and [docs/device_preparation_instructions_windows.md](docs/device_preparation_instructions_windows.md) for Windows.

2. Call the script located at `path/to/aen_acquisition_tool/crack_detector/inference_scripts/crackDetector_remote.py` from your code. 


## Testing in native Python

The crack detector can be called using a Python script as follows.

1. Load the required Python environment.

  open a new terminal windows
  ```bash 
  $ cd path/to/aen_acquisition_tool/crack_detector/inference_scripts
  $ source ~/localInstalls/set_python2.7.sh # for Python 3.7 use source ~/localInstalls/set_python3.7.sh
  ```
2. Run the test script
  ```bash
  $ python test_app.py test_image.bmp ../trained_models/light_0
  ```

## Testing with C++

1. Navigate to the crack_detector directory

  open a new terminal windows
  ```bash
  $ cd path/to/aen_acquisition_tool/crack_detector
  ```

2. Build the binary

   **With Python2.7**
   ```bash
   $ make -f make_and_run_scripts/Makefile_python27 
   ```

   **With Python3.7**
   ```bash
   $ make -f make_and_run_scripts/Makefile_python37 
   ```

3. Execute

   **With Python2.7**: Valid only if built with Python2.7 in the previous step.
   ```bash
   $ ./make_and_run_scripts/run_cpp_test_python27.sh inference_scripts/test_image.bmp light_0 trained_models
   ```

    **With Python3.7**: Valid only if built with Python3.7 in the previous step.
   ```bash
   $ ./make_and_run_scripts/run_cpp_test_python37.sh inference_scripts/test_image.bmp light_0 trained_models
   ```  
