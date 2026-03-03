import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, x_init, y_init, theta_init, map_data):
        self.num_particles = num_particles
        
        # Initialize particles around start pose
        self.particles = np.zeros((num_particles, 3))
        self.particles[:, 0] = x_init + np.random.normal(0, 0.1, num_particles)
        self.particles[:, 1] = y_init + np.random.normal(0, 0.1, num_particles)
        self.particles[:, 2] = theta_init + np.random.normal(0, 0.05, num_particles)
        
        self.weights = np.ones(num_particles) / num_particles
        
        # Map metadata for coordinate conversion
        self.map_info = map_data.info
        self.map_array = np.array(map_data.data).reshape((self.map_info.height, self.map_info.width))
        
        # VAUL Motion Constants
        self.sigma_v = 0.15 
        self.sigma_w = 0.15

    def predict(self, odom_data, dt):
        #Kinematic Model with Gaussian Noise
        v = odom_data['linear_vel']
        w = odom_data['angular_vel']

        # Add noise based on velocity magnitude (VAUL logic)
        v_noisy = v + np.random.normal(0, self.sigma_v, self.num_particles)
        w_noisy = w + np.random.normal(0, self.sigma_w, self.num_particles)

        # Update pose (Euler Integration)
        self.particles[:, 0] += v_noisy * np.cos(self.particles[:, 2]) * dt
        self.particles[:, 1] += v_noisy * np.sin(self.particles[:, 2]) * dt
        self.particles[:, 2] += w_noisy * dt

    def update(self, scan_ranges, angle_min, angle_increment):
        # Downsample scan for performance
        step = 10 
        sampled_ranges = np.array(scan_ranges[::step])
        angles = angle_min + np.arange(len(scan_ranges))[::step] * angle_increment
        
        # Vectorized weight calculation
        new_weights = np.ones(self.num_particles)
        
        for i in range(self.num_particles):
            # Convert particle pose to map coordinates
            px, py, ptheta = self.particles[i]
            
            # Calculate where the LiDAR beams hit for this particle
            hit_x = px + sampled_ranges * np.cos(ptheta + angles)
            hit_y = py + sampled_ranges * np.sin(ptheta + angles)
            
            # Convert hits to grid indices
            grid_x = ((hit_x - self.map_info.origin.position.x) / self.map_info.resolution).astype(int)
            grid_y = ((hit_y - self.map_info.origin.position.y) / self.map_info.resolution).astype(int)
            
            # Filter hits that fall outside the map
            valid = (grid_x >= 0) & (grid_x < self.map_info.width) & \
                    (grid_y >= 0) & (grid_y < self.map_info.height)
            
            # VAUL Evaluation: Likelihood increases if hit land on occupied cells (100)
            hits = self.map_array[grid_y[valid], grid_x[valid]]
            # 100 = Wall, 0 = Free, -1 = Unknown
            # Simplified VAUL Likelihood:
            match_score = np.sum(hits == 100) / len(sampled_ranges)
            new_weights[i] = match_score + 1e-5 # Basic score

        self.weights *= new_weights
        self.weights /= (np.sum(self.weights) + 1e-300)

    def resample(self):
        if 1.0 / np.sum(np.square(self.weights)) < self.num_particles / 2.0:
            indices = np.random.choice(np.arange(self.num_particles), 
                                       size=self.num_particles, 
                                       p=self.weights)
            self.particles = self.particles[indices]
            # Add small jitter to prevent particle deprivation
            self.particles[:, 0] += np.random.normal(0, 0.02, self.num_particles)
            self.particles[:, 1] += np.random.normal(0, 0.02, self.num_particles)
            self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        # VAUL returns the weighted mean or the best particle
        return np.average(self.particles, weights=self.weights, axis=0)
