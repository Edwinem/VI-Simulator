/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2019 Patrick Geneva
 * Copyright (C) 2019 Kevin Eckenhoff
 * Copyright (C) 2019 Guoquan Huang
 * Copyright (C) 2019 OpenVINS Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <iomanip>
#include <cmath>
#include <limits>
#include <iostream>

#include <vio_sim.h>
#include <common_ops.h>


int main(int argc, char **argv) {

  using namespace vi_sim;

  SimParams params;
  params.freq_cam = 2;
  params.trajectory_file = "../data/udel_gore.txt";

  Simulator sim(params);

  const int window_size = 800;


  double time_cam = sim.get_start_time();
  std::vector<int> camids;
  std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> feats;
  Eigen::Matrix<double, 17, 1> imustate;
  Eigen::Matrix<double, 8, 1> pose_state;

  Eigen::Matrix4d gt_pose = Eigen::Matrix4d::Identity();
  bool first_cam_pose = false;
  double start_time = sim.get_start_time();

  double time_imu;
  std::clock_t start;
  double duration;
  /* Your algorithm here */

  int counter = 0;

  Vec3d gyro, acc;

  start = std::clock();


  while (sim.ok()) {
    // IMU: get the next simulated IMU measurement if we have it
    double time_imu;
    Eigen::Vector3d wm, am;
    bool hasimu = sim.get_next_imu(time_imu, wm, am);
    if(hasimu) {
      cout << "new imu measurement = " << std::setprecision(15) << time_imu << std::setprecision(3) << " | w = " << wm.norm() << " | a = " << am.norm() << endl;
    }

    // CAM: get the next simulated camera uv measurements if we have them
    double time_cam;
    std::vector<int> camids;
    std::vector<std::vector<std::pair<size_t,Eigen::VectorXf>>> feats;
    bool hascam = sim.get_next_cam(time_cam, camids, feats);
    if(hascam) {
      cout << "new cam measurement = " << std::setprecision(15) << time_cam;
      cout << std::setprecision(3) << " | " << camids.size() << " cameras | uvs(0) = " << feats.at(0).size() << std::endl;
    }

  }

  double dt = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  std::cout <<  "\n" << dt  << std::endl;

  return 0;
}