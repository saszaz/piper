/**
 *  @file   steap_interface.h
 *  @brief  ROS interface between STEAP and a real/sim robot
 *  @author Mustafa Mukadam
 *  @date   Dec 13, 2016
 **/

#include <steap_interface.h>


namespace piper {

/* ************************************************************************** */
STEAPInterface::STEAPInterface(ros::NodeHandle nh)
{
  // first load problem and setup trajectory client
  problem_ = Problem(nh);
  traj_ = Traj(nh);
  
  // robot state subscriber
  if (nh.hasParam("robot/arm_state_topic"))
  {
    nh.getParam("robot/arm_state_topic", arm_state_topic_);
    arm_state_sub_ = nh.subscribe(arm_state_topic_, 1, &STEAPInterface::armStateCallback, this);
    arm_pos_ = gtsam::Vector::Zero(problem_.robot.getDOFarm());
    arm_pos_time_ = ros::Time::now();
  }
  // robot joint-move client (for start position only)
  nh.getParam("robot/joint_move_service", joint_move_service_);
  joint_move_client_= nh.serviceClient<wam_common::JointMove>(joint_move_service_);

  ros::Duration(1.0).sleep();

  // get start from measurement if not passed as param
  if (!nh.hasParam("start_conf"))
  {
    problem_.start_conf = arm_pos_;
    if (problem_.robot.isThetaNeg())
      problem_.robot.negateTheta(problem_.start_conf);
  }

  // move to trajectory start
  if (nh.getParam("start_conf", start_conf_))
  {
    joint_move_srv_.request.joints = start_conf_;
    ROS_INFO("Moving to start position.");
    joint_move_client_.call(joint_move_srv_);
  }

  // initialize trajectory
  traj_.initializeTrajectory(init_values_, problem_);

  // solve for initial plan with batch gpmp2
  ROS_INFO("Optimizing...");
  int DOF = problem_.robot.getDOF();
  batch_values_ = gpmp2::BatchTrajOptimize3DArm(problem_.robot.arm, problem_.sdf, problem_.start_conf, 
      gtsam::Vector::Zero(DOF), problem_.goal_conf, gtsam::Vector::Zero(DOF), init_values_, problem_.opt_setting);
  ROS_INFO("Batch optimization complete.");

  // publish trajectory for visualization or other use
  if (traj_.plan_traj_pub)
    traj_.publishPlannedTrajectory(batch_values_, problem_, 0);

  // set up incremental inference
  ROS_INFO("Initializing incremental inference...");

  arm_inc_inf_ = gpmp2::ISAM2TrajOptimizer3DArm(problem_.robot.arm, problem_.sdf, problem_.opt_setting);
  arm_inc_inf_.initFactorGraph(problem_.start_conf, gtsam::Vector::Zero(DOF), problem_.goal_conf, gtsam::Vector::Zero(DOF));
  arm_inc_inf_.initValues(batch_values_);
  arm_inc_inf_.update();
  inc_inf_values_ = arm_inc_inf_.values();

  ROS_INFO("Online incremental inference ready.");
}

/* ************************************************************************** */
void STEAPInterface::execute()
{
  size_t exec_step;
  double coll_cost;
  gtsam::Matrix sensor_model;
  int DOF_arm = problem_.robot.getDOFarm();
  int DOF = problem_.robot.getDOF();
  gtsam::Vector conf;
  gtsam::Pose2 pose;

  // sensor model for measurements
  sensor_model = problem_.robot.sensor_arm_sigma*gtsam::Matrix::Identity(DOF_arm, DOF_arm);

  // solve and execute STEAP problem
  ROS_INFO("Executing STEAP online...");

  ros::Time time_start = ros::Time::now();

  for (size_t step=0; step<problem_.total_step-1; step++)
  {
    // interpolate updated solution to a desired resolution for control until next step and check for collision
    exec_values_ = gpmp2::interpolateArmTraj(inc_inf_values_, problem_.opt_setting.Qc_model, problem_.delta_t, 
        problem_.control_inter, step, step+1);
    coll_cost = gpmp2::CollisionCost3DArm(problem_.robot.arm, problem_.sdf, exec_values_, problem_.opt_setting);
    
    
    if (coll_cost != 0)
    {
      ROS_FATAL_STREAM("At step = "<<step<<", plan is not collision free! Collision cost = "<<coll_cost);
      sigintHandler(0);
    }


    // execute trajectory
    ROS_INFO_STREAM("Executing trajectory for step: "<< step << 
      ". Time = "<< (ros::Time::now()-time_start).toSec() << " sec");

    exec_step = problem_.control_inter + 2;
    traj_.executeTrajectory(exec_values_, problem_, exec_step);
    
    // Hold until next timestep
    while ( (ros::Time::now()-time_start).toSec() < (step+1)*problem_.delta_t ) {}

    // get current state and use if it was measured recently then
    // update factor graph to perform incremental inference
    if ((ros::Time::now() - arm_pos_time_).toSec() < 5)
    {
      conf = arm_pos_;  
      if (problem_.robot.isThetaNeg())
        problem_.robot.negateTheta(conf);
        // update
      arm_inc_inf_.addPoseEstimate(step+1, conf, sensor_model);
      arm_inc_inf_.update();
      inc_inf_values_ = arm_inc_inf_.values();
    }
    

    // publish trajectory for visualization or other use
    if (traj_.est_traj_pub)
      traj_.publishEstimatedTrajectory(inc_inf_values_, problem_, step+1);
    if (traj_.plan_traj_pub)
      traj_.publishPlannedTrajectory(inc_inf_values_, problem_, step+1);
  }
}

/* ************************************************************************** */
void STEAPInterface::armStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
  size_t index;
  for (size_t i=0; i<problem_.robot.getDOFarm(); i++)
  {
    index = std::distance(msg->name.begin(), find(msg->name.begin(), msg->name.end(), 
      traj_.arm_joint_names[i]));
    arm_pos_[i] = msg->position[index];
  }
  arm_pos_time_ = ros::Time::now();
}

/* ************************************************************************** */
void STEAPInterface::baseStateCallback(const geometry_msgs::Pose::ConstPtr& msg)
{
  base_pos_ = gtsam::Pose2(msg->position.x, msg->position.y, gtsam::Rot3::Quaternion(msg->orientation.w, 
    msg->orientation.x, msg->orientation.y, msg->orientation.z).yaw());
  base_pos_time_ = ros::Time::now();
}

} // piper namespace


/* ************************************************************************** */
/* main callback */
void mainCallback(const std_msgs::Bool::ConstPtr& msg)
{
  ros::NodeHandle nh("piper");
  piper::STEAPInterface steap(nh);
  double t_start = ros::Time::now().toSec();
  steap.execute();
  ROS_INFO_STREAM("Total execution time: "<<(ros::Time::now().toSec()-t_start));
  ROS_INFO("Done.");  
  ros::shutdown();
}

/* ************************************************************************** */
/* main function */
int main(int argc, char** argv)
{
  ros::init(argc, argv, "steap_interface");
  signal(SIGINT, piper::sigintHandler);
  ros::MultiThreadedSpinner spinner(0);

  ros::NodeHandle n;
  ros::Publisher main_pub = n.advertise<std_msgs::Bool>("/piper/run_main", 1);
  ros::Subscriber main_sub = n.subscribe("/piper/run_main", 1, mainCallback);
  main_pub.publish(std_msgs::Bool());

  spinner.spin();
}
