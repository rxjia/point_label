# Print tag boards

print "out/tag.pdf" in A4 page without scale

# Setup

```
cd 
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone xxxxxx point_label
cd point_label && pip install -r requirements.txt

cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
```

# Run
## 1. launch camera
```
roslaunch realsense2_camera rs_camera.launch align_depth:=true color_width:=1920 color_height:=1080 color_fps:=30
```
## 2. calib relative pose of boards
```
python3 src/get_board_param.py
```
press any key to exit

## 3. run main program
```
python3 src/main_ui.py
```