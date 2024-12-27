# **TurtleBot3 Navigation and Reinforcement Learning in Gazebo**

This repository contains Jupyter notebooks and scripts to simulate, train, and evaluate reinforcement learning algorithms for TurtleBot3 navigation in a Gazebo environment.

---

## **Project Overview**
1. **Reinforcement Learning**:
   - Implements algorithms like Conservative Q-Learning (CQL) for offline training.
   - Evaluates behavior and policy models in a grid-world navigation task.
   - Simulates trajectories and computes success metrics.

2. **Gazebo Simulation**:
   - Scripts in the `navigator` directory control TurtleBot3 in a simulated Gazebo environment.
   - Includes functionality for path planning and obstacle navigation.

3. **Visualization**:
   - Provides trajectory visualizations, evaluation metrics, and debugging tools.

---

## **Getting Started**

### **Prerequisites**
Ensure the following requirements are installed before testing:
- Python 3.8 or higher
- Jupyter Notebook
- Dependencies:
  - `numpy`
  - `torch`
  - `matplotlib`
  - `pandas`
  - `Pillow`
  - `scipy`
  - `optuna`
  - `tqdm`
  - `sklearn`
- Gazebo Simulation Environment (for running TurtleBot3)
- TurtleBot3 ROS Packages (ensure you have `turtlebot3` and `turtlebot3_simulations` installed)
  
To install Python dependencies, run:
```bash
pip install -r requirements.txt
