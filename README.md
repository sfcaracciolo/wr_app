# WR App (Wistar Rat application)
A desktop application to design Wistar Rat (WR) beats from ECG parameters and synthetize them from a ECG generator described on [A Dynamical Model for Generating Synthetic
Electrocardiogram Signals](https://ieeexplore.ieee.org/document/1186732).

## Installation

```
pip install git+https://github.com/sfcaracciolo/wr_app.git
```

## Usage
Run `python src\wr_app\App.py` to launch the application. 

The main window allows the design of the Wistar rat beat from ECG parameters and the tachogram characteristics.

![Main window](/screenshots/main.png)

After synthetizing, a new window is opened to add noise and respiration drift. The save button exports the ECG to *.mat file.

![Main window](/screenshots/viewer.png)
