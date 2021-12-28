# object_detection
Examples and tutorials for object detection algorithms.

# train

Machine: Deep Learning AMI (Ubuntu 18.04) Version 26.0 OS + GPU hardware (https://confluence.check24.de/display/SHFA/AWS+Repository). 

## YOLO

### annotation

This part can be done independent to this repository.

1. Activate virtual environment
2. `pip install labelImg`
3. Launch app:

        labelImg
    or
        
        labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

    with [PRE-DEFINED CLASS FILE] which has the same form as train/yolo/example/classes.names and [IMAGE_PATH] path to the directory of images.

4. Below the 'Save' in the left column, change save format to yolo.
5. command + w: create a bounding box
6. Either select a class from the dropdown or create a new class. The new class will be added to [PRE-DEFINED CLASS FILE]
7. command + s: save the annotation as a txt file.
8. After finishing the annotation. Make a copy of [PRE-DEFINED CLASS FILE] as classes.names in [IMAGE_PATH]

### Installing in Windows

tutorial reference: https://www.youtube.com/watch?v=FE2GBeKuqpc

#### 事前安裝

- CMAKE
- Anaconda
- Visual Studio
- CUDA
- CUDNN
- OPENCV

下載 cmake (cmake.org/download/)

安裝 cmake (win64-x64.msi)


Visual Studio 安裝

    1. 下載和安裝 visual studio community 2019版本
	2. 在安裝時 點選 desktop development with C++
	3. 安裝結束後重新啟動電腦   

CUDA 在 visual studio 安裝好後安裝
CUDA 資料夾位置記好。

安裝OPENCV
tutorial reference: https://www.youtube.com/watch?v=tjXkW0-4gME
創建一個資料夾 build 放在哪裡記好

1. 下載 opencv4.4.0 <sources>
2. 去 github opencv/opencv_contrib
3. 選擇分支 4.4.0
4. 點選 code (綠色，有下載符號). 下載為 .zip檔 (步驟12要用到) 解壓縮 ，位置記好
5. 執行 cmake-gui.exe  https://cmake.org/runningcmake/
6. 在 where is the source code 選擇之前下載好的 opencv4.4.0
7. 在 where to build the binaries 選擇之前創建的 build
  8.  按下 configure
  9.  安裝完後 在 search 輸入 WITH_CU, 然後在 WITH_CUDA 的框框打勾
 10. 在 search 輸入 OPENCV_DN, 然後在BUILD_opencv_dnn, OPENCV_DNN_CUDA, OPENCV_DNN_OPNECL 這三個框框打勾
 11. 在 search 輸入 ENABLE_FA, 然後在ENABLE_FAST_MATH 的框框打勾
 12. 在 search 輸入 OPENCV_EX 然後在 OPENCV_EXTRA_MODULES_PATH 右邊的 value中 選擇之前下載的 opencv_contrib-4.4.0 裡的 module
 
搜尋 WORLD 然後在 BUILD_opencv_world  

 13. 點選 CUDA_FAST_MATH 的框框
 14 CUDA_FAST_MATH 上面有一個 CUDA_ARCH_BIN。在 en.wikipedia.org/wiki/CUDA 中有一張表, 顯示了 Compute capability - GPU 之間的對應，就你的 GPU記好你的 Compute capability，在CUDA_ARCH_BIN右邊的數字中只留下你的GPU對應的 Compute capability. 
 15.  再按一次 Configure
 16 安裝成功後，按下 Generate
  17 在你的 build 資料夾下應該會有一堆東西。像是OPENCV.sln 
  18 以管理員的身份打開 Command Prompt, 進入 build資料夾中，輸入 OPENCV.sln。Visual Studio 會開啟這個檔。若出現了 報錯 “One or more projects in the solution were not loaded correctly…” 不要急。點選 ok。在 Visual Studio SDK中 點選 Tool -> Options -> Project and Solutions -> Web Projects. 把Automatically show data connection… 的選項勾掉 (不要打勾勾)。點選 OK。關上 Visual Studio。再次打開 OPENCV.sln。應該就正常了。
   19 在上方選單欄中把Debug 換成 Release (在 x64 旁) 。
   20 右側應該有一個 panel, 裡面有一票資料夾。打開CmakeTargets，右鍵點選ALL_BUILD。然後點選 build。
   21 這個過程會花半小時以上。 可能會有警告，但無所謂，只要不報錯都好。
   22 成功後，在 ALL_BUILD的下方有一個 INSTALL, 右鍵點選，然後點選 build。這會將opencv安裝在你的系統中。

安裝 DARKNET 

去AlexeyAB/Darknet, 點選 code(綠色，有下載符號)，下載為 .zip
1. 解壓縮，位置記好
2. 解壓縮後裡面有一個資料夾 darknet-master，裡面有一個資料夾 darknet/darknet-master/build，點進去。裡面有一個資料夾darknet，點進去。再點進去x64這個資料夾。打開你的CUDA資料夾，某版本資料夾 v.數字，點進去，然後再點進去bin。目前的位置會像是CUDA/v10.1/bin。裡面有一個檔案cudnn64_7.dll，拷貝。回到剛剛打開的darknet/x64資料夾，貼上。
3. 在你的build資料夾下，點進install 這個資料夾，點進x64，再點進vc16，再點進bin。 目前位置會像是build/install/x64/vc16/bin 拷貝 opencv_world440.dll 這個檔案。回到darknet/x64資料夾 貼上
4.  在darknet資料夾中，有一個檔案 darknet.vcxproj 。用文字編輯器打開。
5. 用control+f 搜尋 CUDA 10，把他們通通改成你的CUDA版本。譬如你用的是CUDA 10.1, 就把 10 改成 10.1
6. 在darknet資料夾中，有一個檔案 yolo_cpp_dll.vcxproj 。用文字編輯器打開。
7. 跟步驟 5 一樣。
8. 點開 yolo_cpp_dll.vcxproj 進入 Visual Studio SDK。
9. 再一次把Debug 換成 Release，同opencv 安裝步驟19
10. 在右邊的panel中有一個 yolo_cpp_dll，右鍵點選。然後點build。沒有錯誤的話，關閉視窗。
11. 這次點開 darknet.sln。同樣把Debug 換成 Release。
12. 右鍵點選右邊panel中的darknet，點選property。
13. Configuration Properties -> C/C++ -> General
14. 把build/install/include這個資料夾的全路徑拷貝下來。
15. 右邊的 Additional Include Directories 選單欄中點選 edit，點選新增資料夾的符號，貼上剛剛拷貝的路徑名稱。按下OK，然後按下Apply。
16. 這次換成 Configuration Properties -> C/C++ -> Preprocessor
17. 在 Preprocessor Definitions 移除 CUDNN_HALF。OK -> Apply
18. Configuration Properties -> CUDA C/C++ -> DEVICE
19. Code Generation 移除compute_75,sm_75。OK -> Apply
20. Configuration Properties -> Linker -> General
21. 先把 build/install/x64/vc16/lib 的全路徑拷貝下來。
22. Additional_Library_Directories 新增資料夾，貼上剛剛拷貝的路徑
23. 右鍵點選darknet，點選build
24. 在darknet/darknet-master/build/darknet/x64 中應該會出現一個 darknet.exe

### training

1. `git clone https://github.com/AlexeyAB/darknet` to get the SoTA repo.

2. In the Makefile, set GPU=1, CUDNN=1, OPENCV=1

3. At line 117 'ifeq ($(GPU), 1) COMMON+= -DGPU -I/usr/local/cuda/include/', replace "/usr/local/cuda/include/" with /usr/local/cuda-11.0/include/

4. Download pre-trained weights depending on the original .cfg file you use. For example, yolov3.cfg and yolov3-spp.cfg will take darknet53.conv.74.

5. Modify the .cfg file. https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

    - In the yolo layers, modify the 'classes' to the number of classes you have.
    
    - In the convolutional layers, modify the 'filters' to (n_of_classes + 5) * n_of_mask, specified in yolo layers.
    
    - Change line max_batches to (classes*2000, but not less than number of training images and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
    
    - Change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
    
    - If the GPU memory is not large enough, increase 'subdivisions' in net layers

6. In the training data directory, type the following linux command lines:
    
    - `find $PWD -name '*train*.jpg' > train.txt`
    
    - `find $PWD -name '*val*.jpg' > test.txt`

7. Modify the .data file: 
    
    - specify the locations of the train.txt and test.txt files generated in the last step in train and valid, respectively.
    
    - backup specifies the location of where the trained models will be saved
    
    - specify the location of the classes.names file

8. In the darkent, compile the files: Make -f Makefile.

9. `./darknet detector train <path to .data> <path to .cfg> <path to pretrained weight>`

### Evaluation

Multiple models will be generated during the training, you can use the validation data specified in .data to find the optimized model:

`./darknet detector map <path to .data> <path to .cfg> <path to trained model> > evluation_report.txt`

## Mask RCNN

### annotation

tutorial: https://www.youtube.com/watch?v=-3WVSxNLk_k

software:
    https://www.robots.ox.ac.uk/~vgg/software/via/

1. Add files: add the images you want to annotate.
2. Create new attribute name in the `Attributes/Region Attributes` if you want to create new attribute.
3. You can have as many attributes as you want on the same object.
3. Select the type as dropdown
4. In the `Id/description` section, you can add new classes as you want. The naming of id/description does not matter.
5. In the `Region Shape` section choose the polygon shape and start your annotation.
6. When you finish all your annotations, `Project`/`Save` so you can resume the project in the future.
7. `Annotation`/`Export Annotations (as json)` to export the annotation. Put the annoation in the folder of the image.


### training


#### Requirement:

- folders:
    train/Mask_RCNN
- files:
    -  config/app.config.yml
    -  algorithms/io/\__init\__.py
    -  algorithms/io/metadata_definition.py
    -  algorithms/io/path_definition.py
    -  algorithms.settings
    
1. Setup your environment with environment.yml    
2. Put your training data in the repository
3. Move under the repository 
4. command line:
       
       python -m train/Mask_RCNN/template/template train --dataset=<relative path to training data> --weights=coco --backbone=resnet101 --object_classification=object
