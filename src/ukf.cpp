#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

    std_a_ = 0.9;
    
    std_yawdd_ = 0.6;
    std_radr_ = 0.3;
    
    std_radphi_ = 0.003;
    
    std_radrd_ = 0.3;
    
    is_initialized_ = false;
    
    n_x_ = 5;
    
    n_aug_ = 7;
    
    lambda_ = -4;
    
    // initial state vector
    x_ = VectorXd(5);
    x_aug = VectorXd(7);
    
    
    // initial covariance matrix
    P_ = MatrixXd(5, 5);
    P_ << 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1000, 0, 0,
    0, 0, 0, 100, 0,
    0, 0, 0, 0, 1;
    
    P_aug = MatrixXd(7, 7);
    
    weights_ = VectorXd(2 * n_aug_ + 1);
    weights_.segment(1, 2 * n_aug_).fill(0.5 / (n_aug_ + lambda_));
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    
    //create sigma point matrix
    Xsig_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    
    n_z = 3;
    Zsig_ = MatrixXd(n_z, 2 * n_aug_ + 1);
    
    
    H_laser_ = MatrixXd(2, 5);
    H_laser_ << 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0;
    
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << 0.0766, 0,
    0, 0.0766;
    
    R_radar_ = MatrixXd(n_z, n_z);
    R_radar_ << std_radr_ * std_radr_, 0, 0,
    0, std_radphi_ * std_radphi_, 0,
    0, 0, std_radrd_ * std_radrd_;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
        if (!is_initialized_) {
            // first measurement
            double x, y;
            
            if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
                x = measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]);
                y = measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]);
           } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
                x = measurement_pack.raw_measurements_[0];
                y = measurement_pack.raw_measurements_[1];
            }
            x_ << x, y, 0, 0, 0;
            
            previous_timestamp_ = measurement_pack.timestamp_;
            is_initialized_ = true;
            cout <<"initialize";
            return;
        }
        
        /*****************************************************************************
         *  Prediction
         ****************************************************************************/
        
        float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
        previous_timestamp_ = measurement_pack.timestamp_;
        
         
        Prediction(dt);
        
        /*****************************************************************************
         *  Update
         ****************************************************************************/
    
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            UpdateRadar(measurement_pack);
            
        } else {
            UpdateLidar(measurement_pack);
            
        }
    
    
}

void UKF::Prediction(double delta_t) {
    AugmentedSigmaPoints();

    SigmaPointPrediction(delta_t);

    PredictMeanAndCovariance();
 

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    VectorXd z_pred = H_laser_ * x_;
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
    MatrixXd S = H_laser_ * P_ * H_laser_.transpose() + R_laser_;
    MatrixXd K = P_ * H_laser_.transpose() * S.inverse();
    
    x_ = x_ + (K * z_diff);
    P_ = P_ - K * S * K.transpose();
    
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;


}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    PredictRadarMeasurement();
    NIS_radar_ = UpdateState(meas_package);
}

void UKF::AugmentedSigmaPoints() {
    x_aug << x_.array(), 0, 0;
    
    
    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;

    
    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();
    Xsig_.colwise() = x_aug;
    
    MatrixXd offset = L * sqrt(lambda_ + n_aug_);
    
    Xsig_.block(0, 1, n_aug_, n_aug_) += offset;
    Xsig_.block(0, n_aug_ + 1, n_aug_, n_aug_) -= offset;

}

void UKF::SigmaPointPrediction(double delta_t) {
    for (int i = 0; i< 2*n_aug_+1; i++)
    {
        //extract values for better readability
        double p_x = Xsig_(0,i);
        double p_y = Xsig_(1,i);
        double v = Xsig_(2,i);
        double yaw = Xsig_(3,i);
        double yawd = Xsig_(4,i);
        double nu_a = Xsig_(5,i);
        double nu_yawdd = Xsig_(6,i);
        
        double px_p, py_p;
        
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }
        
        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;
        
        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;
        
        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;
        
        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }
}

void UKF::PredictMeanAndCovariance() {
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ = x_+ weights_(i) * Xsig_pred_.col(i);
    }
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }
    
}

void UKF::PredictRadarMeasurement() {
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        
        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);
        
        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;
        
        double r = sqrt(p_x * p_x + p_y * p_y);
        double phi = atan2(p_y, p_x);
        double r_dot = (p_x * v1 + p_y * v2) / r;
        
        if (r != r) {
            r = 0;
        }
        if (phi != phi) {
            phi = 0;
        }
        if (r_dot != r_dot) {
            r_dot = 0;
        }
        
        // measurement model
        Zsig_(0,i) = r;                        //r
        Zsig_(1,i) = phi;                                 //phi
        Zsig_(2,i) = r_dot;   //r_dot
    }
    
    //mean predicted measurement
    z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig_.col(i);
    }
    
    //measurement covariance matrix S
    S_ = MatrixXd(n_z, n_z);
    S_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig_.col(i) - z_pred;
        
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        
        S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
    }
    
    //add measurement noise covariance matrix
    S_ = S_ + R_radar_;
}

double UKF::UpdateState(MeasurementPackage meas_package) {
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        
        //residual
        VectorXd z_diff = Zsig_.col(i) - z_pred;
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    
    //Kalman gain K;
    MatrixXd K = Tc * S_.inverse();
    
    //residual
    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_diff = z - z_pred;
    
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    
    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S_*K.transpose();
    
    return z_diff.transpose() * S_.inverse() * z_diff;

}
