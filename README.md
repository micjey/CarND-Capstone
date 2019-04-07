# Capstone Project - Programming a Real Self-Driving Car
**The Udacity self-driving car**
![Carla](https://github.com/bdschrisk/CarND-Capstone-Project/raw/master/imgs/udacity-carla.jpg)
### Team Member:
 - Michael Karg | Team Lead and Developer | Email: micjey@gmx.de
 
When I started Capstone project, there was no team with open positions. So I joined Team 4 (Team Lead: Weston Smith)    which was already full. I worked together with them on different topics but created my own repository. I took over some parts on which I was working in this team and added missing parts of the nodes on myself. So this is a individual submission.

### Implementation of nodes

Using the Robot Operating System (ROS) three nodes had to be developed for controlling a truly autonomous vehicle.
I decided to work in offered workspace because this was supported by Udacity and because of the full functional environment. Only following commands were necessary to build and execute project:

```
sudo apt-get update
sudo apt-get install -y ros-kinetic-dbw-mkz-msgs
cd /home/workspace/CarND-Capstone/ros
rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y

pip uninstall catkin_pkg
pip install -U catkin_pkg==0.4.10
```

To execute the simulation there is a shell script (run.sh) to do all necessary steps to build the project and to start simulation. The most time I struggled with latency problems. As long as I deactivated the camera it was ok but with camera I could not test the whole lap. I improved speed by different adaptions but this was not sufficient. So lets describe the necessary nodes in more detail.

 -  Perception - Traffic Light Detection Node
    Perception means to sense the environment and to perceive for example traffic lights. The traffic light detector was implemented by deep learning approach. Therefore I used a SSD network based on Googles SSD inception model. Because of different trainings data in simulation and real world I decided to use two models for these two environments. The network is loaded at startup and inference is only executed if the car is in the near range of a traffic light. The predictions are accumulated and only published if it is stable for a certain time.

-  Planning - Waypoint Updater
    Planning means the route planning to a given goal state using data from localisation, perception and environment maps. This was implemented in Waypoint Updater. Therefore a snippet of some waypoints are calculated in each cycle. If traffic lights are detected then this node decelerates the speed and ensures that waypoints slow down the vehicle. 
 -  Control - DBW Node
    Control means actualising trajectories formed as part of planning, in order actuate the vehicle, through steering, throttle and brake commands. This was implemented in DBW node by using different controllers like yaw controller, pid and lowpass filter. To ensure correct behavior after test driver did take over the control it is necessary to reset all controllers.


**ROS Node Architecture**
![Node architecture](https://github.com/bdschrisk/CarND-Capstone-Project/raw/master/docs/final-project-ros-graph-v2.png)



### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

