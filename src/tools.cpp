#include "tools.h"
#include "config.h"
#include <iostream>

using Eigen::VectorXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
  // This code is basically right from the Udacity lectures.
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0; // init

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  unsigned long estimation_size = estimations.size();
  unsigned long ground_truth_size = ground_truth.size();

  if (estimation_size != ground_truth_size || estimation_size == 0) {
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
    return rmse;
  }

  //accumulate squared residuals
  for (unsigned int i = 0; i < estimation_size; ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse / estimation_size;

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}