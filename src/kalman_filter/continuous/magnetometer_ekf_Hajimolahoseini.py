import numpy as np

from kalman_filter.discrete_ekf import EKF
# from kalman_filter.continuous.ekf import EKF

class EKF_Hajimolahoseini(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params=model_params)
        # self._fx = np.array([self._x[3]*(self._x[0]*np.cos(self._x[2]*self._dt)-self._x[1]*np.sin(self._x[2]*self._dt)),
        #                     self._x[3]*(self._x[0]*np.sin(self._x[2]*self._dt)+self._x[1]*np.sin(self._x[2]*self._dt)),
        #                     0.0,
        #                     0.0])
        self._F = np.array([[self._x[3]*np.cos(self._x[2]*self._dt),
                             -self._x[3]*np.sin(self._x[2]*self._dt),
                             self._x[3]*self._dt*(-self._x[1]*np.cos(self._dt*self._x[2])-self._x[0]*np.sin(self._dt*self._x[2])),
                             self._x[0] * np.cos(self._x[2] * self._dt) - self._x[1] * np.sin(self._x[2] * self._dt)
                             ],

                            [self._x[3] * np.sin(self._x[2] * self._dt),
                             self._x[3] * np.cos(self._x[2] * self._dt),
                             self._x[3] * self._dt * (self._x[0] * np.cos(self._dt * self._x[2]) - self._x[1]*np.sin(
                                 self._dt * self._x[2])),
                             self._x[0]*np.sin(self._x[2]*self._dt)+self._x[1]*np.sin(self._x[2]*self._dt)
                             ],
                            [0.0, 0.0, 0., 0.],
                            [0., 0., 0., 0.]])
        self._fx = [model_params.A*(np.cos(self._x[2]*self._t)*np.cos(self._x[2]*self._dt)-np.sin(self._x[2]*self._t)*np.sin(self._x[2]*self._dt)),
                        model_params.A*(np.cos(self._x[2]*self._t)*np.sin(self._x[2]*self._dt)+np.sin(self._x[2]*self._t)*np.sin(self._x[2]*self._dt)),
                        self._x[2],
                    self._x[3]]
        self.model_params = model_params

        self._fx1=[model_params.A * (np.cos(model_params.omega * self._t) * np.cos(model_params.omega * self._dt) - np.sin(
            model_params.omega * self._t) * np.sin(model_params.omega * self._dt)),
         model_params.A * (np.cos(model_params.omega * self._t) * np.sin(model_params.omega * self._dt) + np.sin(
             model_params.omega * self._t) * np.sin(model_params.omega * self._dt)),
         model_params.omega, 1.]

    def F(self):
        return np.array([[self._x[3]*np.cos(self._x[2]*self._dt),
                             -self._x[3]*np.sin(self._x[2]*self._dt),
                             self._x[3]*self._dt*(-self._x[1]*np.cos(self._dt*self._x[2])-self._x[0]*np.sin(self._dt*self._x[2])),
                             self._x[0] * np.cos(self._x[2] * self._dt) - self._x[1] * np.sin(self._x[2] * self._dt)
                             ],

                            [self._x[3] * np.sin(self._x[2] * self._dt),
                             self._x[3] * np.cos(self._x[2] * self._dt),
                             self._x[3] * self._dt * (self._x[0] * np.cos(self._dt * self._x[2]) - self._x[1]*np.sin(
                                 self._dt * self._x[2])),
                             self._x[0]*np.sin(self._x[2]*self._dt)+self._x[1]*np.sin(self._x[2]*self._dt)
                             ],
                            [0.0, 0.0, 0., 0.],
                            [0., 0., 0., 0.]])

    def fx(self):
        return [self.model_params.A * (np.cos(self._x[2] * self._t) * np.cos(self._x[2] * self._dt) - np.sin(
            self._x[2] * self._t) * np.sin(self._x[2] * self._dt)),
         self.model_params.A * (np.cos(self._x[2] * self._t) * np.sin(self._x[2] * self._dt) + np.sin(
             self._x[2] * self._t) * np.sin(self._x[2] * self._dt)),
         self._x[2],
         self._x[3]]