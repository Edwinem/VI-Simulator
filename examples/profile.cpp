

#include <iostream>       // std::cout, std::endl
#include <thread>         // std::this_thread::sleep_for
#include <chrono>

#include <vio_sim.h>
#include <common_ops.h>


int main(int argc, char **argv) {

  using namespace vi_sim;

  SimParams params;
  params.freq_cam = 2;
  params.trajectory_file = "../data/udel_gore.txt";

  Simulator sim(params);

  const int window_size = 800;


  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts;

  const auto &map = sim.get_map();

  for (auto iter:map) {
    pts.push_back(iter.second);
  }

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
    bool val = sim.get_next_imu(time_imu, gyro, acc);
    val = sim.get_next_cam(time_cam, camids, feats);
    if (val) {
      first_cam_pose = true;
      val = sim.get_pose(time_cam, pose_state);
      Vec3d p = pose_state.block(5, 0, 3, 1);
      //std::cout << counter++ << "\n " << p << " \n" << std::endl;
      Mat33d Rot = quat_2_Rot(pose_state.block(1, 0, 4, 1));
      gt_pose.block<3, 3>(0, 0) = Rot.transpose();
      gt_pose.block<3, 1>(0, 3) = p;
    }

  }

  double dt = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  std::cout <<  "\n" << dt  << std::endl;

  return 0;
}