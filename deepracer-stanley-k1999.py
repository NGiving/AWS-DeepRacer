import numpy as np

class Reward:
    STEER_MIN = np.radians(-30) # Steering angle min in radians
    STEER_MAX = np.radians(30)  # Steering angle max in radians
    CAR_LENGTH = 0.2            # Length of the car in meters
    k = 0.5                     # Error dampening coefficient
    
    def __init__(self):
        self.all_wheels_on_track = False
        self.is_offtrack = False
        self.theta_c = 0
        self.steering_angle = 0
        self.vc = 0
        self.xc = 0
        self.yc = 0
        self.racing_points = np.array([ [3.05973, 0.68266, 3.65   ],
                                        [3.20951, 0.68313, 3.65   ],
                                        [3.35928, 0.68336, 3.65   ],
                                        [3.50903, 0.6834 , 3.65   ],
                                        [3.65879, 0.68346, 3.65   ],
                                        [3.80856, 0.68352, 3.65   ],
                                        [3.95832, 0.68357, 3.65   ],
                                        [4.10808, 0.68362, 3.65   ],
                                        [4.25783, 0.68367, 3.65   ],
                                        [4.40759, 0.68373, 3.65   ],
                                        [4.55735, 0.68378, 3.65   ],
                                        [4.70711, 0.68384, 3.65   ],
                                        [4.85687, 0.68389, 3.65   ],
                                        [5.00663, 0.68395, 3.65   ],
                                        [5.15639, 0.684  , 3.65   ],
                                        [5.30615, 0.68405, 3.65   ],
                                        [5.45591, 0.68412, 2.83923],
                                        [5.60565, 0.68434, 2.83923],
                                        [5.75542, 0.68429, 1.51309],
                                        [5.9053 , 0.6836 , 1.51309],
                                        [6.05529, 0.68234, 1.51309],
                                        [6.20496, 0.68617, 1.51309],
                                        [6.35406, 0.69852, 1.51201],
                                        [6.50251, 0.71881, 1.51201],
                                        [6.64374, 0.76831, 1.51201],
                                        [6.77489, 0.84127, 1.51201],
                                        [6.89846, 0.92623, 1.51201],
                                        [7.01004, 1.02577, 1.51201],
                                        [7.09975, 1.14609, 1.5305 ],
                                        [7.17247, 1.27703, 1.5305 ],
                                        [7.23045, 1.4172 , 1.5305 ],
                                        [7.27242, 1.56587, 1.5305 ],
                                        [7.28368, 1.71527, 1.53455],
                                        [7.26574, 1.86366, 1.68182],
                                        [7.23396, 2.01073, 1.68182],
                                        [7.1842 , 2.15471, 1.68182],
                                        [7.114  , 2.2871 , 1.68182],
                                        [7.02337, 2.40622, 1.73889],
                                        [6.91743, 2.51266, 1.73889],
                                        [6.79808, 2.60492, 1.73889],
                                        [6.6672 , 2.67759, 1.73889],
                                        [6.52665, 2.72965, 1.73889],
                                        [6.38049, 2.75967, 1.85667],
                                        [6.2298 , 2.77004, 3.15309],
                                        [6.07929, 2.77336, 3.65   ],
                                        [5.92953, 2.77211, 3.65   ],
                                        [5.77978, 2.7708 , 1.96883],
                                        [5.63003, 2.76961, 1.96476],
                                        [5.4803 , 2.76905, 1.75831],
                                        [5.33057, 2.76846, 1.62441],
                                        [5.18075, 2.76536, 1.62441],
                                        [5.03107, 2.76612, 1.62441],
                                        [4.88236, 2.78463, 1.62441],
                                        [4.73518, 2.82126, 1.62441],
                                        [4.59635, 2.879  , 1.62441],
                                        [4.47106, 2.95903, 1.67   ],
                                        [4.3589 , 3.06016, 2.35527],
                                        [4.25573, 3.17014, 2.69864],
                                        [4.16036, 3.28568, 3.65   ],
                                        [4.06673, 3.40247, 3.65   ],
                                        [3.97197, 3.51845, 3.65   ],
                                        [3.87735, 3.63453, 1.83478],
                                        [3.78277, 3.75065, 1.5312 ],
                                        [3.68815, 3.86674, 1.5312 ],
                                        [3.59356, 3.98264, 1.40893],
                                        [3.49883, 4.0995 , 1.24594],
                                        [3.40355, 4.2174 , 1.24594],
                                        [3.29498, 4.31933, 1.24594],
                                        [3.16791, 4.39861, 1.24594],
                                        [3.03874, 4.46137, 1.24594],
                                        [2.85497, 4.49774, 1.24594],
                                        [2.79785, 4.49502, 2.31167],
                                        [2.6333 , 4.49766, 3.65   ],
                                        [2.42942, 4.49807, 3.65   ],
                                        [2.28901, 4.49291, 3.65   ],
                                        [2.14442, 4.48808, 1.52969],
                                        [1.99241, 4.48396, 1.37991],
                                        [1.8428 , 4.47988, 1.37991],
                                        [1.69257, 4.47494, 1.37991],
                                        [1.53988, 4.46866, 1.37991],
                                        [1.38627, 4.45783, 1.37991],
                                        [1.24337, 4.41842, 1.37991],
                                        [1.11356, 4.34595, 1.53878],
                                        [0.99651, 4.25053, 1.53878],
                                        [0.89208, 4.13623, 1.53878],
                                        [0.80509, 4.00657, 1.53878],
                                        [0.74566, 3.86898, 1.53878],
                                        [0.71414, 3.7237 , 1.65611],
                                        [0.70725, 3.57294, 2.18016],
                                        [0.71496, 3.42344, 2.21474],
                                        [0.73656, 3.27569, 2.22102],
                                        [0.77206, 3.12969, 3.6121 ],
                                        [0.81291, 2.98436, 3.65   ],
                                        [0.84943, 2.83849, 3.65   ],
                                        [0.88161, 2.69207, 3.65   ],
                                        [0.91196, 2.54542, 3.65   ],
                                        [0.94235, 2.39877, 3.65   ],
                                        [0.97273, 2.25213, 3.65   ],
                                        [1.00312, 2.10548, 2.58341],
                                        [1.03351, 1.95884, 2.26939],
                                        [1.06385, 1.81222, 1.36364],
                                        [1.09428, 1.66554, 1.36364],
                                        [1.12513, 1.51865, 1.15   ],
                                        [1.15699, 1.37173, 1.15   ],
                                        [1.19869, 1.22808, 1.15   ],
                                        [1.25312, 1.08854, 1.15   ],
                                        [1.33943, 0.96742, 1.15   ],
                                        [1.4401 , 0.85611, 1.15   ],
                                        [1.57205, 0.78639, 1.68646],
                                        [1.71432, 0.73858, 1.99391],
                                        [1.86257, 0.70735, 2.67831],
                                        [2.01155, 0.68592, 2.72456],
                                        [2.16086, 0.67376, 2.72456],
                                        [2.31052, 0.67087, 2.90942],
                                        [2.46047, 0.67614, 3.65   ],
                                        [2.6104 , 0.68087, 3.65   ],
                                        [2.76024, 0.68322, 3.65   ],
                                        [2.90999, 0.68319, 3.65   ]])
        
    # Clamp a value between -30 and 30 degrees of car steering by default
    def clamp(self, val: float, smallest=STEER_MIN, largest=STEER_MAX) -> float:
        return max(smallest, min(val, largest))
    
    # Return the direction the track is facing in radians
    def racing_line_direction(self, prev_point: list, next_point: list) -> float:
        return np.arctan2(next_point[1]-prev_point[1], next_point[0]-prev_point[0])
    
    # Return the index of the previous point and the next point
    def prev_next_racing_point(self) -> list:
        distances = []
        for i in range(len(self.racing_points)):
            distances.append(self.dist_to_point(self.xc, self.yc, self.racing_points[i][0], self.racing_points[i][1]))
        
        closest_index = distances.index(min(distances))
        
        distances_copy = distances.copy()
        distances_copy[closest_index] = 999
        second_closest_index = distances_copy.index(min(distances_copy))
        del distances_copy
        
        # if the prev point is the end index and the next point is at 0, invert the order since list is cyclical
        if len(distances)-1 in [closest_index, second_closest_index] and 0 in [closest_index, second_closest_index]:
            return [len(distances)-1, 0]
        
        return [min(closest_index, second_closest_index), max(closest_index, second_closest_index)]
    
    # Distance between 2 points
    def dist_to_point(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return np.hypot(x1-x2, y1-y2)
    
    # Shortest distance between the line AB and point C
    def dist_point_and_line(self, A: list, B: list, C: list):
        a = np.asarray(A)
        b = np.asarray(B)
        c = np.asarray(C)
        n, v = b - a, c - a
        t = max(0, min(np.dot(v, n)/np.dot(n, n), 1))
        return np.linalg.norm(c - (a + t*n))
    
    # Limit the angle to [0:2pi]
    def pi_2_pi(self, angle: float) -> float:
        if angle > np.pi:
            return angle - 2.0 * np.pi
        if angle < -np.pi:
            return angle + 2.0 * np.pi
        return angle
    
    def reward_function(self, params):
            self.all_wheels_on_track = params['all_wheels_on_track']
            self.is_offtrack = params['is_offtrack']
            self.theta_c = np.radians(params['heading'])
            self.steering_angle = np.radians(params['steering_angle'])
            self.vc = params['speed']
            self.xc = params['x'] + np.cos(self.theta_c) * self.CAR_LENGTH/2
            self.yc = params['y'] + np.sin(self.theta_c) * self.CAR_LENGTH/2
            
            if self.is_offtrack:
                return float(1e-3)
                
            reward = 1
            
            [prev_index, next_index] = self.prev_next_racing_point()
            prev_p, next_p = self.racing_points[prev_index], self.racing_points[next_index]
            
            # Direction of the track
            theta_track = self.racing_line_direction(prev_p, next_p)
            
            # Heading error
            psi = self.pi_2_pi(theta_track - self.theta_c)
            
            # Cross track error
            e = self.dist_point_and_line(prev_p[:2], next_p[:2], (self.xc, self.yc))
            
            # Steering
            delta = psi + np.arctan2(self.vc, self.k * e)
    
            # Clip between min and max steer angles
            delta = self.clamp(delta)
    
            reward += (np.radians(60) - abs(self.steering_angle - delta)) / np.radians(60)
            
            # Speed difference
            delta_s = abs(prev_p[2] - self.vc)
            
            reward += (1 - (delta_s/4)**2)**2
            
            return float(reward)

reward_obj = Reward()
def reward_function(params):
    return reward_obj.reward_function(params)
    