#pragma once

#include <Eigen/Geometry>
#include <vector>


/**
 * @brief Reimplementations of OpenCV Undistort Functions
 *
 * @section Summary
 * This file contains exact reimplmentations of the OpenCV undistortPoints functions for use with Eigen vectors
 * and matrices.
 */


namespace vi_sim {

using Vec3d=Eigen::Vector3d;
using Vec2d=Eigen::Vector2d;
using Vec4d=Eigen::Vector4d;
using VecXd=Eigen::Matrix<double, Eigen::Dynamic, 1>;

using Mat33d=Eigen::Matrix3d;
using Mat44d=Eigen::Matrix4d;
using MatXd=Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

template<typename Scalar>
using Mat33s=Eigen::Matrix<Scalar, 3, 3>;
template<typename Scalar>
using MatXs=Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template<typename Scalar>
using VecXs=Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template<typename Scalar>
Eigen::Matrix<Scalar, 3, 3> create_skew(const Eigen::Matrix<Scalar, 3, 1> &w) {
  Eigen::Matrix<Scalar, 3, 3> w_x;
  w_x << 0, -w(2), w(1),
      w(2), 0, -w(0),
      -w(1), w(0), 0;
  return w_x;
}

struct termCriteria {
  enum Type {
    COUNT=1,
    MAX_ITER=COUNT,
    EPS=2
  } ;

  int maxCount;
  double epsilon;
  int type;
};


template<typename FLOAT>
void computeTiltProjectionMatrixIMPL(FLOAT tauX,
                                 FLOAT tauY,
                                 Eigen::Matrix<FLOAT, 3, 3> *matTilt = 0,
                                 Eigen::Matrix<FLOAT, 3, 3> *dMatTiltdTauX = 0,
                                 Eigen::Matrix<FLOAT, 3, 3> *dMatTiltdTauY = 0,
                                 Eigen::Matrix<FLOAT, 3, 3> *invMatTilt = 0) {
  FLOAT cTauX = cos(tauX);
  FLOAT sTauX = sin(tauX);
  FLOAT cTauY = cos(tauY);
  FLOAT sTauY = sin(tauY);
  Eigen::Matrix<FLOAT, 3, 3> matRotX;
  matRotX << 1, 0, 0, 0, cTauX, sTauX, 0, -sTauX, cTauX;
  Eigen::Matrix<FLOAT, 3, 3> matRotY;
  matRotY << cTauY, 0, -sTauY, 0, 1, 0, sTauY, 0, cTauY;
  Eigen::Matrix<FLOAT, 3, 3> matRotXY = matRotY * matRotX;
  Eigen::Matrix<FLOAT, 3, 3> matProjZ;
  matProjZ << matRotXY(2, 2), 0, -matRotXY(0, 2), 0, matRotXY(2, 2), -matRotXY(1, 2), 0, 0, 1;
  if (matTilt) {
    // Matrix for trapezoidal distortion of tilted image sensor
    *matTilt = matProjZ * matRotXY;
  }
  if (dMatTiltdTauX) {
    // Derivative with respect to tauX
    Eigen::Matrix<FLOAT, 3, 3> temp1;
    temp1 << 0, 0, 0, 0, -sTauX, cTauX, 0, -cTauX, -sTauX;
    Eigen::Matrix<FLOAT, 3, 3> dMatRotXYdTauX = matRotY * temp1;
    Eigen::Matrix<FLOAT, 3, 3> dMatProjZdTauX;
    dMatProjZdTauX << dMatRotXYdTauX(2, 2), 0, -dMatRotXYdTauX(0, 2),
        0, dMatRotXYdTauX(2, 2), -dMatRotXYdTauX(1, 2), 0, 0, 0;
    *dMatTiltdTauX = (matProjZ * dMatRotXYdTauX) + (dMatProjZdTauX * matRotXY);
  }
  if (dMatTiltdTauY) {
    // Derivative with respect to tauY
    Eigen::Matrix<FLOAT, 3, 3> temp;
    temp << -sTauY, 0, -cTauY, 0, 0, 0, cTauY, 0, -sTauY;
    Eigen::Matrix<FLOAT, 3, 3> dMatRotXYdTauY = temp * matRotX;
    Eigen::Matrix<FLOAT, 3, 3> dMatProjZdTauY;
    dMatProjZdTauY << dMatRotXYdTauY(2, 2), 0, -dMatRotXYdTauY(0, 2),
        0, dMatRotXYdTauY(2, 2), -dMatRotXYdTauY(1, 2), 0, 0, 0;
    *dMatTiltdTauY = (matProjZ * dMatRotXYdTauY) + (dMatProjZdTauY * matRotXY);
  }
  if (invMatTilt) {
    FLOAT inv = 1. / matRotXY(2, 2);
    Eigen::Matrix<FLOAT, 3, 3> invMatProjZ;
    invMatProjZ << inv, 0, inv * matRotXY(0, 2), 0, inv, inv * matRotXY(1, 2), 0, 0, 1;
    *invMatTilt = matRotXY.transpose() * invMatProjZ;
  }
}

template<typename POINT_FLOAT, typename MATRIX_FLOAT>
void undistortPointsIMPL(const std::vector<Eigen::Matrix<POINT_FLOAT, 2, 1>,
                                       Eigen::aligned_allocator<Eigen::Matrix<POINT_FLOAT, 2, 1>>> &input_pts,
                     std::vector<Eigen::Matrix<POINT_FLOAT, 2, 1>,
                                 Eigen::aligned_allocator<Eigen::Matrix<POINT_FLOAT, 2, 1>>> &output_pts,
                     const Mat33s<MATRIX_FLOAT> &K,
                     const VecXs<MATRIX_FLOAT> &dist,
                     const MatXs<MATRIX_FLOAT> &R,
                     const MatXs<MATRIX_FLOAT> &P) {

  termCriteria criteria;
  criteria.type = termCriteria::COUNT;
  criteria.maxCount = 5;
  criteria.epsilon = 0.01;

  bool valid_distoration = false;

  output_pts.resize(input_pts.size());

  Mat33d A, RR;
  VecXd k(14);
  k.setZero();

  Mat33d invMatTilt = Mat33d::Identity();
  Mat33d matTilt = Mat33d::Identity();

  A = K.template cast<double>();

  if (dist.size() > 0) {
    int s = dist.size();
    bool valid = (s == 4 || s == 5 || s == 8 || s == 12 || s == 14);
    assert(valid);
    valid_distoration = valid;
    VecXd temp_k=dist.template cast<double>();
    k.segment(0,temp_k.size())=temp_k;
  }

  if ((k.size()> 12) && (k[12] != 0 || k[13] != 0)) {
    computeTiltProjectionMatrixIMPL<double>(k[12], k[13], NULL, NULL, NULL, &invMatTilt);
    computeTiltProjectionMatrixIMPL<double>(k[12], k[13], &matTilt, NULL, NULL);
  }

  if (R.size() > 0) {
    assert(R.rows() == 3 && R.cols() == 3);
    RR = R.template cast<double>();
  } else {
    RR = Mat33d::Identity();
  }

  if (P.size() > 0) {

    assert((P.rows() == 3 && (P.cols() == 3 || P.cols() == 4)));
    Mat44d P_copy = P.template cast<double>();
    RR = P_copy.block<3, 3>(0, 0) * RR;
  }

  double fx = A(0, 0);
  double fy = A(1, 1);
  double ifx = 1. / fx;
  double ify = 1. / fy;
  double cx = A(0, 2);
  double cy = A(1, 2);

  int n = input_pts.size();
  for (int i = 0; i < n; i++) {
    double x, y, x0 = 0, y0 = 0, u, v;

    x = input_pts[i][0];
    y = input_pts[i][1];
    u = x;
    v = y;
    x = (x - cx) * ifx;
    y = (y - cy) * ify;

    if (valid_distoration) {
      // compensate tilt distortion
      Vec3d vecUntilt = invMatTilt * Vec3d(x, y, 1);
      double invProj = vecUntilt(2) ? 1. / vecUntilt(2) : 1;
      x0 = x = invProj * vecUntilt(0);
      y0 = y = invProj * vecUntilt(1);

      double error = std::numeric_limits<double>::max();
      // compensate distortion iteratively

      for (int j = 0;; j++) {
        if ((criteria.type & termCriteria::COUNT) && j >= criteria.maxCount)
          break;
        if ((criteria.type & termCriteria::EPS) && error < criteria.epsilon)
          break;
        double r2 = x * x + y * y;
        double icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) / (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
        if (icdist < 0)  // test: undistortPoints.regression_14583
        {
          x = (u - cx) * ifx;
          y = (v - cy) * ify;
          break;
        }
        double deltaX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x) + k[8] * r2 + k[9] * r2 * r2;
        double deltaY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y + k[10] * r2 + k[11] * r2 * r2;
        x = (x0 - deltaX) * icdist;
        y = (y0 - deltaY) * icdist;

        if (criteria.type & termCriteria::EPS) {
          double r4, r6, a1, a2, a3, cdist, icdist2;
          double xd, yd, xd0, yd0;
          Vec3d vecTilt;

          r2 = x * x + y * y;
          r4 = r2 * r2;
          r6 = r4 * r2;
          a1 = 2 * x * y;
          a2 = r2 + 2 * x * x;
          a3 = r2 + 2 * y * y;
          cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
          icdist2 = 1. / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
          xd0 = x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
          yd0 = y * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4;

          vecTilt = matTilt * Vec3d(xd0, yd0, 1);
          invProj = vecTilt(2) ? 1. / vecTilt(2) : 1;
          xd = invProj * vecTilt(0);
          yd = invProj * vecTilt(1);

          double x_proj = xd * fx + cx;
          double y_proj = yd * fy + cy;

          error = sqrt(pow(x_proj - u, 2) + pow(y_proj - v, 2));
        }
      }
    }

    double xx = RR(0, 0) * x + RR(0, 1) * y + RR(0, 2);
    double yy = RR(1, 0) * x + RR(1, 1) * y + RR(1, 2);
    double ww = 1. / (RR(2, 0) * x + RR(2, 1) * y + RR(2, 2));
    x = xx * ww;
    y = yy * ww;

    output_pts[i][0] = x;
    output_pts[i][1] = y;

  }

}

inline void rodrigues(const Vec3d &rot_vec, Mat33d &rot_matrix) {
  rot_matrix = Mat33d::Identity();

  const double theta = rot_vec.norm();
  if (theta > 1e-8) {
    double s = sin(theta);
    double c = cos(theta);
    double cs = 1.0 - cos(theta);
    double inv_t = 1.0 / theta;
    const Mat33d skew = create_skew(rot_vec);

    const Mat33d I = Mat33d::Identity();

    double x = rot_vec[0] * inv_t;
    double y = rot_vec[1] * inv_t;
    double z = rot_vec[2] * inv_t;

    Mat33d rrt;
    rrt << x * x, x * y, x * z,
        x * y, y * y, y * z,
        x * z, y * z, z * z;

    rot_matrix = c * I + cs * rrt + s * skew;

  } else {
    rot_matrix = Mat33d::Identity();
  }
}

template<typename POINT_FLOAT, typename MATRIX_FLOAT>
void FishEyeundistortPointsIMPL(const std::vector<Eigen::Matrix<POINT_FLOAT, 2, 1>,
                                              Eigen::aligned_allocator<Eigen::Matrix<POINT_FLOAT,
                                                                                     2,
                                                                                     1>>> &distorted_pts,
                            std::vector<Eigen::Matrix<POINT_FLOAT, 2, 1>,
                                        Eigen::aligned_allocator<Eigen::Matrix<POINT_FLOAT, 2, 1>>> &undistorted_pts,
                            const Mat33s<MATRIX_FLOAT> &K,
                            const VecXs<MATRIX_FLOAT> &D,
                            const MatXs<MATRIX_FLOAT> &R,
                            const MatXs<MATRIX_FLOAT> &P) {

  undistorted_pts.resize(distorted_pts.size());

  if (P.size() > 0) {
    assert((P.rows() == 3 && P.cols() == 3) || (P.rows() == 3 && P.cols() == 4));
  }

  if (R.size() > 0) {
    assert((R.rows() == 3 && R.cols() == 3) || (R.size() == 3));

  }
  assert(D.size() == 4);
  assert(K.rows() == 3 && K.cols() == 3);

  Vec2d f, c;
  Mat33d camMat = K.template cast<double>();

  f = Vec2d(camMat(0, 0), camMat(1, 1));
  c = Vec2d(camMat(0, 2), camMat(1, 2));

  Vec4d k = D;

  Mat33d RR = Mat33d::Identity();
  if (R.size() > 0 && R.size() == 3) {
    Vec3d rvec = R.template cast<double>();

    rodrigues(rvec, RR);
  } else if ((R.size() > 0) && (R.cols() == 3 && R.rows() == 3))
    RR = R.template cast<double>();

  if (P.size() > 0) {
    Mat33d PP;
    Mat44d P_copy=P.template cast<double>();
    PP = P_copy.block<3, 3>(0, 0);
    RR = PP * RR;
  }

  // start undistorting


  size_t n = distorted_pts.size();

  for (size_t i = 0; i < n; i++) {
    Vec2d pi;
    pi[0]= distorted_pts[i][0];  // image point
    pi[1]= distorted_pts[i][1];

    Vec2d pw((pi[0] - c[0]) / f[0], (pi[1] - c[1]) / f[1]);      // world point

    double scale = 1.0;

    double theta_d = sqrt(pw[0] * pw[0] + pw[1] * pw[1]);

    // the current camera model is only valid up to 180 FOV
    // for larger FOV the loop below does not converge
    // clip values so we still get plausible results for super fisheye images > 180 grad
    theta_d = std::min(std::max(double(-EIGEN_PI / 2.), theta_d), double(EIGEN_PI / 2.));

    if (theta_d > 1e-8) {
      // compensate distortion iteratively
      double theta = theta_d;

      const double EPS = 1e-8; // or std::numeric_limits<double>::epsilon();
      for (int j = 0; j < 10; j++) {
        double theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 = theta6 * theta2;
        double k0_theta2 = k[0] * theta2, k1_theta4 = k[1] * theta4, k2_theta6 = k[2] * theta6,
            k3_theta8 = k[3] * theta8;
        /* new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta) */
        double theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
            (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
        theta = theta - theta_fix;
        if (fabs(theta_fix) < EPS)
          break;
      }

      scale = std::tan(theta) / theta_d;
    }

    Vec2d pu = pw * scale; //undistorted point

    // reproject
    Vec3d pr = RR * Vec3d(pu[0], pu[1], 1.0); // rotated point optionally multiplied by new camera matrix
    Vec2d fi(pr[0] / pr[2], pr[1] / pr[2]);       // final


    undistorted_pts[i][0] = fi[0];
    undistorted_pts[i][1] = fi[1];
  }
}


void undistortPoints(const std::vector<Eigen::Matrix<float, 2, 1>,
                                       Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>> &input_pts,
                     std::vector<Eigen::Matrix<float, 2, 1>,
                                 Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>> &output_pts,
                     const Mat33d &K,
                     const VecXd &dist,
                     const MatXd &R=MatXd(),
                     const MatXd &P=MatXd()) {

  undistortPointsIMPL(input_pts, output_pts, K, dist, R, P);
}
void undistortPoints(const std::vector<Vec2d, Eigen::aligned_allocator<Vec2d>> &input_pts,
                     std::vector<Vec2d, Eigen::aligned_allocator<Vec2d>> &output_pts,
                     const Mat33d &K,
                     const VecXd &dist,
                     const MatXd &R,
                     const MatXd &P) {

  undistortPointsIMPL(input_pts, output_pts, K, dist, R, P);
}

void undistortPoints(const Vec2d& input_pts,
                     Vec2d &output_pts,
                     const Mat33d &K,
                     const VecXd &dist,
                     const MatXd &R,
                     const MatXd &P) {
  std::vector<Eigen::Matrix<double, 2, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 2, 1>>> input,output;
  input.push_back(input_pts);
  undistortPointsIMPL(input, output, K, dist, R, P);
  output_pts=output[0];
}
void undistortPoints(const Eigen::Vector2f& input_pts,
                     Eigen::Vector2f &output_pts,
                     const Mat33d &K,
                     const VecXd &dist,
                     const MatXd &R=MatXd(),
                     const MatXd &P=MatXd()) {

  std::vector<Eigen::Matrix<float, 2, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>> input,output;
  input.push_back(input_pts);
  undistortPointsIMPL(input, output, K, dist, R, P);
  output_pts=output[0];
}

void FishEyeundistortPoints(const Eigen::Vector2f &distorted_pts,
                            Eigen::Vector2f &undistorted_pts,
                            const Mat33d &K,
                            const VecXd &D,
                            const MatXd &R=MatXd(),
                            const MatXd &P=MatXd()) {
  std::vector<Eigen::Matrix<float, 2, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>> input,output;
  input.push_back(distorted_pts);
  FishEyeundistortPointsIMPL(input, output, K, D, R, P);
  undistorted_pts=output[0];
}

void FishEyeundistortPoints(const std::vector<Eigen::Matrix<float, 2, 1>,
                                              Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>> &input_pts,
                            std::vector<Eigen::Matrix<float, 2, 1>,
                                        Eigen::aligned_allocator<Eigen::Matrix<float, 2, 1>>> &output_pts,
                            const Mat33d &K,
                            const VecXd &D,
                            const MatXd &R=MatXd(),
                            const MatXd &P=MatXd()) {
  FishEyeundistortPointsIMPL(input_pts, output_pts, K, D, R, P);
}

}
