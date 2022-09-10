use ode_solvers::*;
use rand::{prelude::Distribution, thread_rng};
use statrs::distribution::Normal;
use std::{
    fs::File,
    io::BufWriter,
    io::Write,
    path::Path,
};
use ukf::{Control, Covariance, Output, State, UnscentedKalmanFilter};

/// Save the results of the simulation to a file.
/// 
/// Output format is time,pos,vel,imu,gps,filt_pos,filt_vel
fn save(
    dt: f64,
    times: &Vec<f64>,
    states: &Vec<State<2>>,
    imu: &Vec<Output<1>>,
    gps: &Vec<Output<1>>,
    gps_hz: f64,
    filt: &Vec<(State<2>, Covariance<2>)>,
    filename: &Path,
) {
    let mut time_since_last_gps: f64 = 0.0;
    let mut gps_i: usize = 0;
    // Create or open file
    let file = match File::create(filename) {
        Err(e) => {
            println!("Could not open file. Error: {:?}", e);
            return;
        }
        Ok(buf) => buf,
    };
    let mut buf = BufWriter::new(file);

    for (i, state) in states.iter().enumerate() {
        buf.write_fmt(format_args!("{}", times[i])).unwrap();
        for val in state.iter() {
            buf.write_fmt(format_args!(",{}", val)).unwrap();
        }
        for val in imu[i].iter() {
            buf.write_fmt(format_args!(",{}", val)).unwrap();
        }

        buf.write_fmt(format_args!(",{}", gps[gps_i][0])).unwrap();

        if time_since_last_gps >= 1.0 / gps_hz {
            time_since_last_gps = 0.0;
            gps_i += 1;
        }
        time_since_last_gps += dt;

        for val in filt[i].0.iter() {
            buf.write_fmt(format_args!(",{}", val)).unwrap();
        }

        buf.write_fmt(format_args!("\n")).unwrap();
    }
    if let Err(e) = buf.flush() {
        println!("Could not write to file. Error: {:?}", e);
    }
}

/// Structure to hold the parameters of our dynamics.
#[derive(Clone, Copy)]
struct DoubleIntegratorWithViscosity {
    mu: f64,
    u: Control<1>,
}

/// Implement continuous dynamics numerical integration.
impl ode_solvers::System<State<2>> for DoubleIntegratorWithViscosity {
    fn system(&self, _: f64, y: &State<2>, dy: &mut State<2>) {
        let ydot = self.dynamics_continuous(y, &self.u);
        dy[0] = ydot[0];
        dy[1] = ydot[1];
    }
}

/// Implement measurement models and dynamics.
impl DoubleIntegratorWithViscosity {
    fn imu_measurement_model(&self, x: &State<2>) -> Output<1> {
        let e = -self.mu * x[1] + self.u[0];
        Output::<1>::from_element(e)
    }

    fn gps_measurement_model(&self, x: &State<2>) -> Output<1> {
        Output::<1>::from_element(x[0])
    }

    fn dynamics_continuous(&self, x: &State<2>, u: &Control<1>) -> State<2> {
        let xdot0 = x[1];
        let xdot1 = -self.mu * x[1] + u[0];
        State::<2>::from_vec(vec![xdot0, xdot1])
    }

    fn dynamics_discreate(&self, x: &State<2>, dt: &f64) -> State<2> {
        let mut stepper = Dop853::new(*self, 0.0, *dt, *dt, *x, 1e-4, 1e-4);
        let res = stepper.integrate();

        // Handle result
        match res {
            Ok(_) => {
                *stepper.y_out().last().unwrap()
            }
            Err(e) => {
                println!("An error occured: {}", e);
                State::<2>::zeros()
            }
        }
    }
}

fn main() {
    // Simulation duration
    let dt = 0.01;
    const N: usize = 1000;

    // measurement and process noise standard deviations
    let imu_stdv: f64 = 0.01;
    let gps_stdv: f64 = 0.1;
    let process_noise_stdv = 0.01;

    // GPS subsampling
    let mut gps_i = 0;
    let gps_hz: f64 = 1.0;
    let mut time_since_last_gps: f64 = 0.0;

    // Filter parameters
    let alpha = 1.0;
    let beta = 2.0;
    let kappa = 0.0;

    // Viscosity parameter
    let viscosity = 1.0;

    // Data records
    let mut traj = Vec::<State<2>>::new();
    let mut time: Vec<f64> = Vec::new();
    let mut imu_readings: Vec<Output<1>> = Vec::new();
    let mut gps_readings: Vec<Output<1>> = Vec::new();
    let mut filt_readings: Vec<(State<2>, Covariance<2>)> = Vec::new();

    // Initial values
    let x0 = State::<2>::zeros();
    let p0 = Covariance::<2>::from_diagonal_element(0.01);
    let control = Control::<1>::from_element(1.0);

    // process and measurement covariances
    let process_cov = Covariance::<2>::from_diagonal_element(process_noise_stdv);
    let imu_cov = Covariance::<1>::from_diagonal_element(imu_stdv);
    let gps_cov = Covariance::<1>::from_diagonal_element(gps_stdv);

    // random noise for the state and measurements
    let mut rng = thread_rng();
    let rand_state_noise = Normal::new(0.0, process_noise_stdv).unwrap();
    let rand_imu_noise = Normal::new(0.0, imu_stdv).unwrap();
    let rand_gps_noise = Normal::new(0.0, gps_stdv).unwrap();

    // model of a double integrator with viscosity 
    let model = DoubleIntegratorWithViscosity {
        mu: viscosity,
        u: control,
    };

    // unscented kalman filter
    let mut ukf = UnscentedKalmanFilter::<2>::new(alpha, beta, kappa, x0, p0);

    // measurement and dynamics update closures
    let imu_fn = |x: &State<2>, u: &Control<1>| model.imu_measurement_model(x);
    let gps_fn = |x: &State<2>, u: &Control<1>| model.gps_measurement_model(x);
    let dyns_fn = |x: &State<2>, u: &Control<1>, dt: &f64| model.dynamics_discreate(x, dt);

    // set up state trajectory and sensor readings
    traj.push(x0);
    time.push(0.0);

    imu_readings.push(model.imu_measurement_model(&x0));
    gps_readings.push(model.gps_measurement_model(&x0));

    for i in 1..N {
        let x = model.dynamics_discreate(&traj[i - 1], &dt);
        let noise = State::<2>::from_fn(|_, _| rand_state_noise.sample(&mut rng));
        traj.push(x + noise);
        time.push(i as f64 * dt);

        let imu_noise = Output::<1>::from_fn(|_, _| rand_imu_noise.sample(&mut rng));
        imu_readings.push(model.imu_measurement_model(&x) + imu_noise);

        time_since_last_gps += dt;

        if time_since_last_gps >= 1.0 / gps_hz {
            time_since_last_gps = 0.0;
            let gps_noise = Output::<1>::from_fn(|_, _| rand_gps_noise.sample(&mut rng));
            gps_readings.push(model.gps_measurement_model(&x) + gps_noise)
        }
    }

    time_since_last_gps = 1.0 / gps_hz;

    // filter the state trajectory with the UKF
    for i in 0..N {
        println!("Filter iteration {}", i);
        filt_readings.push((ukf.state, ukf.covariance));
        ukf.innovate(&control, &imu_readings[i], imu_fn, &imu_cov);

        if time_since_last_gps >= 1.0 / gps_hz {
            time_since_last_gps = 0.0;
            ukf.innovate(&control, &gps_readings[gps_i], gps_fn, &gps_cov);
            gps_i += 1;
        }

        time_since_last_gps += dt;

        ukf.update(&control, dt, dyns_fn, &process_cov);
    }

    // save results
    save(
        dt,
        &time,
        &traj,
        &imu_readings,
        &gps_readings,
        gps_hz,
        &filt_readings,
        Path::new("double_integrator_w_vis.csv"),
    );
}
