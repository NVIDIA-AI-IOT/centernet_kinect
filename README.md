# centernet_kinect

This repository demonstrates how to set up [**Azure Kinect** camera] (https://azure.microsoft.com/en-us/services/kinect-dk/) with your Jetson platform, collect and annotate data, train a object ddetection model with the collected data, and finally run a real time object detection model on your development kit. <br/>

* [Install Sensor SDK on Jetson](#install_sensor_sdk)
* [Collect/Annotate Data](#collect_annotate_data)
  * [Example Subsection](#example_subsection)

<a name="install_sensor_sdk"></a>
## Install Sensor SDK on Jetson

**Note** in this tutorial we will be installing the SDK on Ubuntu Version *18.04.5 LTS"*.<br/>

To check your distribution/version you can run the following command
```bash
cat /etc/os-release
```
Here is the [link](https://packages.microsoft.com/) to microsoft package repository in case you are using any other distribution/version.<br/>
Here are more instructions on how to configure and install the SDK on other platforms [Link] (https://docs.microsoft.com/en-us/windows-server/administration/linux-package-repository-for-microsoft-software) <br/>


### 1. Add Microsoft's product repository for ARM64
```bash
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/multiarch/prod
sudo apt-get update
```

### 2. Install Kinect Package
```bash
sudo apt install k4a-tools
sudo apt install libk4a1.4-dev
```

### 3. Setup udev rules
- in order to use the Azure Kinect SDK with the device and without being 'root', you will need to setup udev rules [Link](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#linux-device-setup)

- Copy '[scripts/99-k4a.rules](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/scripts/99-k4a.rules)' into '/etc/udev/rules.d/'.
- Detach and reattach Azure Kinect devices if attached during this process.

### 3. Setup udev rules
- test the SDK
```bash
k4aviewer
```

**Note** Here are more instuctions if you were experiencing dificulty with yout setup [Link](https://gist.github.com/madelinegannon/c212dbf24fc42c1f36776342754d81bc#updating-firmware-for-azure-kinect)

<a name="collect_annotate_data"></a>
## Collect/Annotate Data