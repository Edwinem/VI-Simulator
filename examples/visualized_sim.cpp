#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <iostream>       // std::cout, std::endl
#include <thread>         // std::this_thread::sleep_for
#include <chrono>

#include <vio_sim.h>
#include <common_ops.h>

const u_int8_t cam_color[3]{250, 0, 26};
const u_int8_t state_color[3]{250, 0, 26};
const u_int8_t pose_color[3]{0, 50, 255};
const u_int8_t gt_color[3]{0, 171, 47};

void DrawCameraFrustrum(Eigen::Matrix4d &pose,
                        const float width = .15,
                        const float height = .1,
                        const float depth = .3,
                        const u_int8_t *color = cam_color) {

  pangolin::OpenGlMatrix pl(pose);

  glPushMatrix();

  glMultMatrixd(pl.m);

  const float &w = width;
  const float &h = height;
  const float &z = depth;

  glLineWidth(2);
  glColor3f(0.0f, 1.0f, 0.0f);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(w, h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, h, z);

  glVertex3f(w, h, z);
  glVertex3f(w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(-w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);

  glVertex3f(-w, -h, z);
  glVertex3f(w, -h, z);
  glEnd();

  glPopMatrix();
}

void DrawAxis() {
  //    glLineWidth(30);
//    glBegin(GL_LINES);
//    glColor3f(1.0f,0.0f,0.0f);
//    // X axis
//    glVertex3f(0, 0.0f, 0.0f);
//    glVertex3f(30, 0.0f, 0.0f);
//    // Y axis
//    glColor3f(0.0f,1.0f,0.0f);
//    glVertex3f(0.0f, 0, 0.0f);
//    glVertex3f(0.0f, 30, 0.0f);
//    // Z axis
//    glColor3f(0.0f,0.0f,1.0f);
//    glVertex3f(0.0f, 0.0f, 0);
//    glVertex3f(0.0f, 0.0f, 30);
//
//    glEnd();
}

int main(int argc, char **argv) {

  using namespace vi_sim;

  SimParams params;
  params.freq_cam = 2;
  params.trajectory_file = "../data/udel_gore.txt";

  Simulator sim(params);

  const int window_size = 800;

  pangolin::CreateWindowAndBind("Main", window_size, window_size);

  glEnable(GL_DEPTH_TEST);

  pangolin::OpenGlRenderState camera(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
      pangolin::ModelViewLookAt(-5, 5, -5, 0, 0, 0, pangolin::AxisY));

  pangolin::View &display3D =
      pangolin::CreateDisplay()
          .SetAspect(1)
          .SetBounds(0.0, 1.0, 0.0, 1.0)
          .SetHandler(new pangolin::Handler3D(camera));

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

  start = std::clock();

  /* Your algorithm here */

  int counter = 0;

  Vec3d gyro, acc;

  while (!pangolin::ShouldQuit()) {
    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    display3D.Activate(camera);
    //pangolin::glDrawColouredCube();

    // Render OpenGL Cube
    glPointSize(3);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //pangolin::glDrawPoints(pts);

    if (sim.ok()) {
      bool val = sim.get_next_imu(time_imu, gyro, acc);
      val = sim.get_next_cam(time_cam, camids, feats);
      if (val) {
        first_cam_pose = true;
        val = sim.get_pose(time_cam, pose_state);
        Vec3d p = pose_state.block(5, 0, 3, 1);
        std::cout << counter++ << "\n " << p << " \n" << std::endl;
        Mat33d Rot = quat_2_Rot(pose_state.block(1, 0, 4, 1));
        gt_pose.block<3, 3>(0, 0) = Rot.transpose();
        gt_pose.block<3, 1>(0, 3) = p;
        //DrawCameraFrustrum(gt_pose);
      } else if (first_cam_pose) {
        //DrawCameraFrustrum(gt_pose);
      }

    } else {

    }

//    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
//    double dt=time_cam-start_time;
//    if(duration<(time_cam-start_time)){
    //std::this_thread::sleep_for (std::chrono::duration<double>(0.15));
//      start = std::clock();
//      start_time=time_cam;
//    } else{
//      start = std::clock();
//      start_time=time_cam;
//    }




    // Swap frames and Process Events
    pangolin::FinishFrame();
  }

  return 0;
}